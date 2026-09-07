"""
GHOST v2 — FULL COGNITIVE pipeline over REAL hardware (injection loopback).

Streams a dataset into three ESP32 boards over USB, reads the echoed CSI back,
and runs the whole chain end to end — through the localizer and the AI layer:

    dataset --INJ--> ESP32 x3 (USB) --CSI_DATA--> gateway -> signal cleaner
            -> feature extractor -> localizer -> agentic_core_v2 (LLM)
            -> coordinates/velocity JSON

This is the cognitive counterpart of run_serial_demo.py (which stops at
features). No Wi-Fi. Each board is one USB port; port order = RX1, RX2, RX3.

Usage (venv terminal, boards flashed with csi_inject_serial, no monitors open):
    ./.venv/bin/python v2/run_serial_cognitive_demo.py \
        --ports /dev/cu.usbserial-0001 /dev/cu.usbserial-3 /dev/cu.usbserial-5 \
        --llm --pos-scale embedded_wifi

Drop --llm to use the deterministic passthrough (no Ollama needed). With --llm,
start Ollama first:  docker compose up -d
"""

import argparse
import glob
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import serial


def autodetect_ports() -> list:
    """Find ESP32 USB serial ports on macOS/Linux, sorted for stable RX order.

    Lets the script run with no --ports (e.g. from an IDE run button). Bluetooth
    and debug ports are excluded.
    """
    patterns = ["/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB*",
                "/dev/cu.usbmodem*", "/dev/ttyUSB*", "/dev/ttyACM*"]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    found = sorted(p for p in found if "Bluetooth" not in p and "debug" not in p)
    return found

from v2.config_v2 import DEFAULT_DATASET_PATH, SAMPLE_RATE_HZ
from v2.ghost.gateway_v2 import CSIParser, ComplexMatrix
from v2.ghost.main_v2 import GhostV2Pipeline, _print_decision, add_log_flags, apply_log_flags
from v2.injector.injector import Injector, build_adapter, _NODE_NAMES
from v2.run_serial_demo import SerialFanoutSender


def main() -> int:
    ap = argparse.ArgumentParser(description="GHOST v2 full cognitive pipeline over 3 boards")
    ap.add_argument("--ports", nargs=3, metavar=("RX1", "RX2", "RX3"),
                    help="three serial ports, in RX1 RX2 RX3 order "
                         "(auto-detected if omitted)")
    ap.add_argument("--path", default=DEFAULT_DATASET_PATH, help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--calib", type=int, default=100, help="calibration preamble frames")
    ap.add_argument("--frames", type=int, default=200, help="operational frames")
    ap.add_argument("--rate", type=float, default=8.0, help="frames/sec (keep low for 115200)")
    ap.add_argument("--window", type=int, default=100, help="localization window (samples)")
    ap.add_argument("--breath-window", type=int, default=0, dest="breath_window",
                    help="breathing window (samples; needs many frames). 0 disables.")
    ap.add_argument("--step", type=int, default=50, help="window hop between decisions")
    ap.add_argument("--pos-scale", default="embedded_wifi", dest="pos_scale")
    ap.add_argument("--llm", action="store_true", help="route decisions through Ollama")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    if not os.path.exists(args.path):
        print(f"dataset not found: {args.path}")
        return 1

    port_names = args.ports
    if not port_names:
        detected = autodetect_ports()
        if len(detected) < 3:
            print(f"auto-detect found {len(detected)} serial port(s): {detected}")
            print("need exactly 3 ESP32 boards. Plug in all three, or pass "
                  "--ports RX1 RX2 RX3 explicitly.")
            return 1
        port_names = detected[:3]
        print(f"auto-detected ports (RX1 RX2 RX3 = {port_names})")
        if len(detected) > 3:
            print(f"  note: {len(detected)} ports present; using the first 3. "
                  "Pass --ports to choose the order explicitly.")

    ports = [serial.Serial(p, args.baud, timeout=1) for p in port_names]
    port_by_node = {name: ports[i] for i, name in enumerate(_NODE_NAMES)}
    print(f"opened {port_names} — waiting for boards to boot...")
    time.sleep(2.0)
    for p in ports:
        p.reset_input_buffer()

    total = args.calib + args.frames
    matrix = ComplexMatrix(max_time=total + 50)
    parser = CSIParser()
    stop = threading.Event()

    def reader(rx_index: int, ser: serial.Serial):
        while not stop.is_set():
            try:
                raw = ser.readline()
            except (OSError, serial.SerialException):
                break
            if not raw:
                continue
            packet = parser.parse(raw.decode("utf-8", "replace").strip(), rx_index)
            if packet is not None:
                matrix.append(rx_index, packet.csi)

    threads = [threading.Thread(target=reader, args=(i, ports[i]), daemon=True)
               for i in range(3)]
    for t in threads:
        t.start()

    adapter = build_adapter(args.dataset, args.path)
    injector = Injector(adapter, sender=SerialFanoutSender(port_by_node),
                        rate_hz=args.rate, calibration_samples=args.calib)
    print(f"streaming {args.calib} calib + up to {args.frames} operational frames "
          f"at {args.rate} fps (~{total / args.rate:.0f}s)...")
    injector.run(max_frames=args.frames)

    prev, stable = -1, 0
    for _ in range(200):
        time.sleep(0.1)
        got = min(matrix.get_receiver_count(r) for r in range(3))
        stable = stable + 1 if got == prev else 0
        prev = got
        if stable >= 5 and got > 0:
            break
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    for p in ports:
        p.close()

    counts = [matrix.get_receiver_count(r) for r in range(3)]
    received = min(counts)
    print(f"\nEchoed CSI_DATA received per board: {counts}  (sent {total})")
    if received <= args.calib:
        print("  not enough frames came back to separate calibration from data")
        return 1

    window = matrix.get_latest(received)
    pipe = GhostV2Pipeline(pos_scale=args.pos_scale, non_metric=True, use_llm=args.llm)
    pipe.calibrate(window[:, :, : args.calib])
    op = window[:, :, args.calib:]
    n_op = op.shape[2]
    dt = args.step / float(SAMPLE_RATE_HZ)

    print(f"\n=== Cognitive decisions over REAL hardware "
          f"(window={args.window}, step={args.step}, llm={'on' if args.llm else 'off'}) ===")
    first = None
    for i, start in enumerate(range(0, max(1, n_op - args.window + 1), args.step)):
        op_slice = op[:, :, start:start + args.window]
        breath_slice = None
        if args.breath_window and start + args.breath_window <= n_op:
            breath_slice = op[:, :, start:start + args.breath_window]
        decision, features, *_ = pipe.process(op_slice, dt=dt if i > 0 else 0.0,
                                              breath_slice=breath_slice)
        _print_decision(decision, features)
        if first is None:
            first = decision

    if first is not None:
        print("\n=== full JSON of the first decision ===")
        print(json.dumps(first.to_dict(), indent=2))
    print("\nFull cognitive hardware demo complete "
          "(dataset -> 3x ESP32 -> gateway -> cleaner -> features -> localizer -> LLM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
