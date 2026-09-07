"""
GHOST v2 — full pipeline over REAL hardware, injected through USB serial.

Streams a dataset into three ESP32 boards over their USB cables and runs the
echoed CSI_DATA back through the whole v2 pipeline:

    dataset --INJ--> ESP32 x3 (USB) --CSI_DATA--> complex matrix
            --> signal cleaner (static subtraction) --> feature extractor

No Wi-Fi. Each board is one USB port; port order = RX1, RX2, RX3.

Usage (venv terminal, boards flashed with csi_inject_serial, no monitors open):
    ./.venv/bin/python v2/run_serial_demo.py \
        --ports /dev/cu.usbserial-3 /dev/cu.usbserial-0001 /dev/cu.usbserial-XXXX
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import serial

from v2.config_v2 import DEFAULT_DATASET_PATH
from v2.ghost.gateway_v2 import CSIParser, ComplexMatrix
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2
from v2.ghost.feature_extractor_v2 import FeatureExtractorV2
from v2.ghost.main_v2 import add_log_flags, apply_log_flags
from v2.injector.injector import Injector, build_adapter, _NODE_NAMES


def iq_to_inj_line(seq: int, iq: np.ndarray) -> bytes:
    """Serialize one complex frame as an 'INJ,seq,v0,...,v127' line."""
    i = np.clip(np.rint(iq.real), -32768, 32767).astype(int)
    q = np.clip(np.rint(iq.imag), -32768, 32767).astype(int)
    interleaved = np.empty(iq.shape[0] * 2, dtype=int)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return ("INJ," + str(seq) + "," + ",".join(map(str, interleaved.tolist())) + "\n").encode()


class SerialFanoutSender:
    """Drop-in replacement for the injector's UDP sender: writes INJ lines to
    the serial port assigned to each node."""

    def __init__(self, port_by_node: dict):
        self._ports = port_by_node

    def send(self, node_name: str, seq: int, iq: np.ndarray, calibration: bool = False) -> None:
        self._ports[node_name].write(iq_to_inj_line(seq, np.asarray(iq)))
        self._ports[node_name].flush()

    def close(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="GHOST v2 serial hardware pipeline demo")
    ap.add_argument("--ports", nargs=3, required=True, metavar=("RX1", "RX2", "RX3"),
                    help="three serial ports, in RX1 RX2 RX3 order")
    ap.add_argument("--path", default=DEFAULT_DATASET_PATH, help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--calib", type=int, default=100, help="calibration preamble frames")
    ap.add_argument("--frames", type=int, default=200, help="operational frames")
    ap.add_argument("--rate", type=float, default=8.0, help="frames/sec (keep low for 115200)")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    if not os.path.exists(args.path):
        print(f"dataset not found: {args.path}")
        return 1

    ports = [serial.Serial(p, args.baud, timeout=1) for p in args.ports]
    port_by_node = {name: ports[i] for i, name in enumerate(_NODE_NAMES)}
    print(f"opened {args.ports} — waiting for boards to boot...")
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
            line = raw.decode("utf-8", "replace").strip()
            packet = parser.parse(line, rx_index)
            if packet is not None:
                matrix.append(rx_index, packet.csi)

    threads = [threading.Thread(target=reader, args=(i, ports[i]), daemon=True) for i in range(3)]
    for t in threads:
        t.start()

    adapter = build_adapter(args.dataset, args.path)
    sender = SerialFanoutSender(port_by_node)
    injector = Injector(adapter, sender=sender, rate_hz=args.rate,
                        calibration_samples=args.calib)
    print(f"streaming {args.calib} calib + up to {args.frames} operational frames "
          f"at {args.rate} fps (this takes ~{total / args.rate:.0f}s)...")
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

    counts = [matrix.get_receiver_count(r) for r in range(3)]
    received = min(counts)
    print(f"\nEchoed CSI_DATA received per board: {counts}  (sent {total})")
    for p in ports:
        p.close()
    if received <= args.calib:
        print("  not enough frames came back to separate calibration from data")
        return 1

    window = matrix.get_latest(received)
    calib_slice = window[:, :, : args.calib]
    op_slice = window[:, :, args.calib:]

    cleaner = SignalCleanerV2()
    cleaner.calibrate_from(calib_slice)
    cleaned = cleaner.clean(op_slice)
    features = FeatureExtractorV2().extract(cleaned)

    print("\n=== Signal cleaning (static subtraction) ===")
    print(f"  raw operational   |H_raw|     mean : {float(np.mean(np.abs(op_slice))):8.3f}")
    print(f"  after subtraction |H_dynamic| mean : {float(np.mean(np.abs(cleaned.dynamic))):8.3f}")
    print(f"  bandpassed motion  A_clean    std  : {float(np.std(cleaned.amplitude)):8.3f}")

    print("\n=== Extracted features ===")
    print(f"  breathing_frequency : {features.breathing_frequency:.3f} Hz")
    print(f"  total_energy        : {features.total_energy:.2f}")
    print(f"  doppler_mean        : {features.doppler_mean:.3f} Hz")
    print(f"  variance_rx         : {[round(v, 3) for v in features.variance_rx]}")
    print(f"  node_energies       : "
          + ", ".join(f"{k}={v:.2f}" for k, v in features.node_energies.items()))

    print("\nReal-hardware pipeline demo complete "
          "(dataset -> 3x ESP32 over USB -> gateway -> cleaner -> features).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
