"""
feature_extractor_test.py — Live test of the full GHOST sensing pipeline.

Connects to ESP32 receivers via the Gateway, captures real CSI data,
runs it through SignalCleaner → FeatureExtractor, and exports state
vectors to CSV.

Usage:
    python feature_extractor_test.py                        # 30s capture
    python feature_extractor_test.py -d 60                  # 60s capture
    python feature_extractor_test.py -o features.csv        # custom output
    python feature_extractor_test.py -p /dev/cu.usbserial-0001 /dev/cu.usbserial-0002 /dev/cu.usbserial-0003
"""

import csv
import sys
import time
import logging
import argparse

from gateway import Gateway
from signal_cleaner import SignalCleaner
from feature_extractor import FeatureExtractor, FEATURE_NAMES

FS = 100.0                         # ESP32 packet rate (Hz)
WINDOW_SEC = 2.0                   # extraction window (seconds)
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 200
EXTRACT_INTERVAL = 0.5             # extract every 0.5 seconds
WARMUP_SEC = 3.0                   # wait for buffer to fill

DEFAULT_OUTPUT = "test1.csv"
DEFAULT_DURATION = 30.0            # capture duration (seconds)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Live feature extraction from ESP32 CSI receivers"
    )
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                    help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("-d", "--duration", type=float, default=DEFAULT_DURATION,
                    help=f"Capture duration in seconds (default: {DEFAULT_DURATION})")
    ap.add_argument("-p", "--ports", nargs="+", default=None,
                    help="Serial port paths (auto-detect if omitted)")
    ap.add_argument("--fs", type=float, default=FS,
                    help=f"Sampling frequency Hz (default: {FS})")
    args = ap.parse_args()

    fs = args.fs
    window_samples = int(fs * WINDOW_SEC)

    # ------------------------------------------------------------------
    # 1. Start Gateway
    # ------------------------------------------------------------------
    gw = Gateway(ports=args.ports)
    gw.start()

    if not gw._readers:
        print("No receivers connected. Exiting.")
        sys.exit(1)

    n_active = len(gw._readers)
    print(f"\nGateway started — {n_active} receiver(s) connected")
    print(f"Capture: {args.duration}s  |  Window: {WINDOW_SEC}s  |  Interval: {EXTRACT_INTERVAL}s")
    print(f"Output: {args.output}\n")

    # ------------------------------------------------------------------
    # 2. Warm up — wait for buffer to fill
    # ------------------------------------------------------------------
    matrix = gw.get_matrix()
    print(f"Warming up ({WARMUP_SEC}s) — waiting for receivers to fill buffer...")

    warmup_start = time.time()
    while time.time() - warmup_start < WARMUP_SEC:
        time.sleep(0.5)
        counts = [matrix.get_receiver_count(i) for i in range(n_active)]
        print(f"  Packets received: {counts}")

    min_count = min(matrix.get_receiver_count(i) for i in range(n_active))
    if min_count < window_samples:
        print(f"  Warning: only {min_count} packets buffered "
              f"(need {window_samples} for a full window).")
        print(f"  Continuing — early windows may contain zero-padded data.\n")
    else:
        print(f"  Buffer ready ({min_count} packets).\n")

    # ------------------------------------------------------------------
    # 3. Init pipeline components
    # ------------------------------------------------------------------
    cleaner = SignalCleaner(fs=fs)
    extractor = FeatureExtractor(fs=fs, window_samples=window_samples)

    # ------------------------------------------------------------------
    # 4. Capture loop — extract features every EXTRACT_INTERVAL seconds
    # ------------------------------------------------------------------
    rows = []
    capture_start = time.time()
    extract_count = 0

    print("Extracting features... Press Ctrl+C to stop early.\n")
    print(f"{'time':>6s}  {'breath_Hz':>9s}  {'energy':>10s}  {'doppler':>9s}  "
          f"{'var_rx1':>9s}  {'var_rx2':>9s}  {'var_rx3':>9s}")
    print("-" * 68)

    try:
        while time.time() - capture_start < args.duration:
            # Grab the latest 2-second window from the live buffer.
            raw = matrix.get_latest(window_samples)  # (3, 64, window_samples)

            # Run the pipeline: clean → extract.
            cleaned = cleaner.clean(raw)
            state = extractor.extract(cleaned)  # (6,)

            elapsed = time.time() - capture_start
            extract_count += 1

            # Print live.
            print(f"{elapsed:6.1f}  {state[0]:9.4f}  {state[1]:10.2f}  {state[2]:9.4f}  "
                  f"{state[3]:9.4f}  {state[4]:9.4f}  {state[5]:9.4f}")

            # Store for CSV.
            rows.append([f"{elapsed:.2f}"] + [f"{v:.6f}" for v in state])

            # Gateway stats every 10 extractions.
            if extract_count % 10 == 0:
                stats = gw.get_stats()
                for label, s in stats.items():
                    total = s["received"] + s["lost"]
                    loss = (s["lost"] / total * 100) if total > 0 else 0
                    print(f"  >> {label}: {s['received']} pkts, "
                          f"{s['lost']} lost ({loss:.1f}%), "
                          f"{s['errors']} errors")

            time.sleep(EXTRACT_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # ------------------------------------------------------------------
    # 5. Stop gateway and write CSV
    # ------------------------------------------------------------------
    gw.stop()

    if not rows:
        print("No data captured.")
        sys.exit(1)

    header = ["time_sec"] + FEATURE_NAMES
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"  Captured {len(rows)} state vectors over {args.duration:.0f}s")
    print(f"  Written to: {args.output}")
    print(f"  Columns: {header}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()