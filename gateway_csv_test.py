"""
gateway_csv_test.py — Capture live CSI data from ESP32 receivers and write
a CSV file compatible with csiTest.py.

CSV structure (no header, matches csiTest.py expectations):
    Columns 0-22 : standard ESP32 CSI metadata fields (type through len)
    Column  23   : elapsed time in seconds (float)       <- TIME_COL
    Column  24   : first_word
    Column  25   : raw CSI I/Q array, e.g. "[1,2,3,...,128]"  <- CSI_COL

Usage:
    python gateway_csv_test.py                          # auto-detect, 30s
    python gateway_csv_test.py --duration 60            # capture 60 seconds
    python gateway_csv_test.py -o my_data.csv           # custom output file
    python gateway_csv_test.py -p /dev/cu.usbserial-0001
"""

import csv
import sys
import time
import argparse
import logging

import serial

from gateway import (
    Gateway,
    CSIParser,
    AmplitudeMatrix,
    DEFAULT_BAUD_RATE,
    SERIAL_TIMEOUT,
)

logger = logging.getLogger("ghost.gateway_test")

OUTPUT_FILE = "example_csi.csv"
DEFAULT_DURATION = 30  # seconds


def parse_raw_line(line: str):
    """Split a raw CSI_DATA serial line into metadata fields + CSI data string.

    Returns (meta_fields, first_word, csi_str) or None for non-CSI / bad lines.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None

    bracket_start = line.find("[")
    bracket_end = line.rfind("]")
    if bracket_start == -1 or bracket_end == -1 or bracket_end <= bracket_start:
        return None

    csi_str = line[bracket_start:bracket_end + 1]

    # Metadata is everything before the bracket array.
    meta_part = line[:bracket_start].rstrip('," ')
    fields = meta_part.split(",")

    # Standard ESP32 format has 25 fields (indices 0-24).
    # We need at least 23 (type through len) plus first_word.
    if len(fields) < 23:
        return None

    meta = fields[:23]  # columns 0-22: type, id, mac, ... , len
    first_word = fields[23] if len(fields) > 23 else "0"
    return meta, first_word, csi_str


def capture_csi(port: str, duration: float, output: str, baud_rate: int):
    """Read CSI data from one serial port and write to CSV."""
    parser = CSIParser()
    matrix = AmplitudeMatrix()

    try:
        ser = serial.Serial(port=port, baudrate=baud_rate, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        sys.exit(1)

    start_time = time.time()
    count = 0

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)

        print(f"Capturing CSI data from {port} -> {output}")
        print(f"Duration: {duration}s  |  Baud: {baud_rate}")
        print("Press Ctrl+C to stop early.\n")

        try:
            while time.time() - start_time < duration:
                raw = ser.readline()
                if not raw:
                    continue

                try:
                    line = raw.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    continue

                parsed = parse_raw_line(line)
                if parsed is None:
                    continue

                meta, first_word, csi_str = parsed
                elapsed = time.time() - start_time

                # Build row: meta(0-22), time(23), first_word(24), csi_data(25)
                row = meta + [f"{elapsed:.6f}", first_word, csi_str]
                writer.writerow(row)
                count += 1

                # Feed into amplitude matrix for live stats
                packet = parser.parse(line, receiver_index=0)
                if packet:
                    matrix.append(0, packet.amplitude)

                if count % 50 == 0:
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"  {count} packets  |  {elapsed:.1f}s  |  {rate:.0f} pkt/s")

        except KeyboardInterrupt:
            print("\nStopped by user.")

    ser.close()
    total = time.time() - start_time
    print(f"\nDone. {count} CSI packets saved to {output}")
    if total > 0 and count > 0:
        print(f"  Avg rate: {count / total:.1f} packets/sec")
    print(f"\nRun csiTest.py to visualize:\n  python csiTest.py")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Capture CSI data from ESP32 to CSV (for csiTest.py)"
    )
    ap.add_argument("-p", "--port", help="Serial port path (auto-detect if omitted)")
    ap.add_argument("-o", "--output", default=OUTPUT_FILE,
                    help=f"Output CSV path (default: {OUTPUT_FILE})")
    ap.add_argument("-d", "--duration", type=float, default=DEFAULT_DURATION,
                    help=f"Capture duration in seconds (default: {DEFAULT_DURATION})")
    ap.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD_RATE,
                    help=f"Baud rate (default: {DEFAULT_BAUD_RATE})")
    args = ap.parse_args()

    port = args.port
    if not port:
        gw = Gateway()
        ports = gw.detect_ports()
        if not ports:
            print("No ESP32 serial ports detected. Use -p to specify manually.")
            sys.exit(1)
        port = ports[0]
        print(f"Auto-detected port: {port}")

    capture_csi(port, args.duration, args.output, args.baud)


if __name__ == "__main__":
    main()