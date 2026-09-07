"""
Serial-injection smoke test (no Wi-Fi).

One process owns the USB serial port: it writes INJ frames to the ESP32 and
reads the echoed CSI_DATA lines back. Proves the USB loopback end to end.

Flash the csi_inject_serial firmware first, and make sure no monitor/screen is
holding the port. Then:

    ./.venv/bin/python v2/serial_test.py /dev/cu.usbserial-0001
    ./.venv/bin/python v2/serial_test.py /dev/cu.usbserial-0001 --count 50
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import serial


def main() -> None:
    ap = argparse.ArgumentParser(description="USB serial injection smoke test")
    ap.add_argument("port", help="serial port, e.g. /dev/cu.usbserial-0001")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--count", type=int, default=10, help="frames to send")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"opened {args.port} @ {args.baud} — waiting for the board to boot...")
    time.sleep(2.0)
    ser.reset_input_buffer()

    received = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                raw = ser.readline()
            except (OSError, serial.SerialException):
                break
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("CSI_DATA"):
                received.append(line)
                print("  <<", line[:70], "...")

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    print(f"sending {args.count} INJ frames...")
    for i in range(args.count):
        iq = np.arange(128, dtype=int)
        line = "INJ," + str(i) + "," + ",".join(str(int(v)) for v in iq) + "\n"
        ser.write(line.encode())
        ser.flush()
        time.sleep(0.15)

    time.sleep(1.0)
    stop.set()
    reader_thread.join(timeout=2.0)
    ser.close()

    print(f"\nDONE: sent {args.count}, received {len(received)} CSI_DATA lines back")
    if received:
        print("first echoed line:\n ", received[0])
    else:
        print("no CSI_DATA came back — see troubleshooting notes")


if __name__ == "__main__":
    main()
