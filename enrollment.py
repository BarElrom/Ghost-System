"""
enrollment.py — Fingerprint enrollment tool for the GHOST memory bank.

Connects to ESP32 receivers (live mode) or reads a previously captured
CSV (offline mode), runs the Layer 2 pipeline, and stores labeled state
vectors in ChromaDB.

Usage:
    # Live enrollment from ESP32 receivers:
    python enrollment.py --zone Kitchen --activity Static --duration 30
    python enrollment.py --zone Bedroom --activity Walking --duration 60

    # Offline enrollment from a CSV exported by feature_extractor_test.py:
    python enrollment.py --from-csv test1.csv --zone Kitchen --activity Static

    # List all enrolled zones:
    python enrollment.py --list-zones

    # Clear the collection:
    python enrollment.py --clear
"""

import csv
import sys
import time
import logging
import argparse

import numpy as np

from memory_bank import MemoryBank
from feature_extractor import FeatureExtractor, FEATURE_NAMES, NUM_FEATURES
from signal_cleaner import SignalCleaner

FS = 100.0                             # ESP32 packet rate (Hz)
WINDOW_SEC = 2.0                       # extraction window (seconds)
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 200
EXTRACT_INTERVAL = 0.5                 # extract every 0.5 seconds
WARMUP_SEC = 3.0                       # wait for buffer to fill

DEFAULT_DURATION = 30.0                # capture duration (seconds)

logger = logging.getLogger("ghost.enrollment")


def enroll_from_csv(
    csv_path: str,
    zone: str,
    activity: str,
    label: str,
    bank: MemoryBank,
) -> int:
    """Read state vectors from a CSV file and enroll them in the memory bank.

    The CSV must have a header row with columns matching FEATURE_NAMES,
    as produced by feature_extractor_test.py.

    Returns the number of fingerprints enrolled.
    """
    vectors = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vec = np.array(
                    [float(row[name]) for name in FEATURE_NAMES],
                    dtype=np.float64,
                )
                vectors.append(vec)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping row: %s", e)
                continue

    if not vectors:
        print(f"No valid state vectors found in {csv_path}")
        return 0

    zones = [zone] * len(vectors)
    activities = [activity] * len(vectors)
    labels = [label] * len(vectors) if label else None

    ids = bank.enroll_batch(vectors, zones, activities, labels)
    return len(ids)


def enroll_live(
    zone: str,
    activity: str,
    label: str,
    duration: float,
    ports: list[str] | None,
    bank: MemoryBank,
) -> int:
    """Capture live CSI data and enroll state vectors in the memory bank.

    Returns the number of fingerprints enrolled.
    """
    # Import Gateway here to avoid requiring serial hardware for CSV mode.
    from gateway import Gateway

    gw = Gateway(ports=ports)
    gw.start()

    if not gw._readers:
        print("No receivers connected. Exiting.")
        return 0

    n_active = len(gw._readers)
    matrix = gw.get_matrix()

    print(f"Gateway started — {n_active} receiver(s)")
    print(f"Warming up ({WARMUP_SEC}s)...")

    warmup_start = time.time()
    while time.time() - warmup_start < WARMUP_SEC:
        time.sleep(0.5)
        counts = [matrix.get_receiver_count(i) for i in range(n_active)]
        print(f"  Packets received: {counts}")

    cleaner = SignalCleaner(fs=FS)
    extractor = FeatureExtractor(fs=FS, window_samples=WINDOW_SAMPLES)

    vectors = []
    capture_start = time.time()

    print(f"\nEnrolling: zone='{zone}' activity='{activity}' for {duration}s")
    print(f"{'time':>6s}  {'breath_Hz':>9s}  {'energy':>10s}  {'doppler':>9s}  "
          f"{'var_rx1':>9s}  {'var_rx2':>9s}  {'var_rx3':>9s}")
    print("-" * 68)

    try:
        while time.time() - capture_start < duration:
            raw = matrix.get_latest(WINDOW_SAMPLES)
            cleaned = cleaner.clean(raw)
            state = extractor.extract(cleaned)

            elapsed = time.time() - capture_start
            vectors.append(state)

            print(f"{elapsed:6.1f}  {state[0]:9.4f}  {state[1]:10.2f}  "
                  f"{state[2]:9.4f}  {state[3]:9.4f}  {state[4]:9.4f}  "
                  f"{state[5]:9.4f}")

            time.sleep(EXTRACT_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    gw.stop()

    if not vectors:
        print("No data captured.")
        return 0

    zones = [zone] * len(vectors)
    activities = [activity] * len(vectors)
    labels = [label] * len(vectors) if label else None

    ids = bank.enroll_batch(vectors, zones, activities, labels)
    return len(ids)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Enroll fingerprints into the GHOST memory bank"
    )
    ap.add_argument("--zone", type=str,
                    help="Zone label (e.g., Kitchen, Bedroom, Hall)")
    ap.add_argument("--activity", type=str,
                    help="Activity label (e.g., Static, Walking, Breathing)")
    ap.add_argument("--label", type=str, default="",
                    help="Optional human-readable description")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                    help=f"Capture duration in seconds (default: {DEFAULT_DURATION})")
    ap.add_argument("--from-csv", type=str, default=None,
                    help="Path to a CSV file with state vectors (offline mode)")
    ap.add_argument("-p", "--ports", nargs="+", default=None,
                    help="Serial port paths for live mode (auto-detect if omitted)")
    ap.add_argument("--list-zones", action="store_true",
                    help="List all enrolled zones and exit")
    ap.add_argument("--clear", action="store_true",
                    help="Clear all fingerprints from the collection")
    ap.add_argument("--count", action="store_true",
                    help="Print fingerprint count and exit")
    args = ap.parse_args()

    bank = MemoryBank()

    # --- utility commands ---
    if args.list_zones:
        zones = bank.get_all_zones()
        print(f"Enrolled zones ({len(zones)}): {zones}")
        print(f"Total fingerprints: {bank.count()}")
        return

    if args.count:
        print(f"Total fingerprints: {bank.count()}")
        return

    if args.clear:
        count_before = bank.count()
        bank.clear()
        print(f"Cleared {count_before} fingerprints from collection")
        return

    # --- enrollment commands ---
    if not args.zone or not args.activity:
        print("Error: --zone and --activity are required for enrollment.")
        print("Example: python enrollment.py --zone Kitchen --activity Static")
        sys.exit(1)

    count_before = bank.count()

    if args.from_csv:
        print(f"Enrolling from CSV: {args.from_csv}")
        print(f"  Zone: {args.zone}  Activity: {args.activity}")
        n = enroll_from_csv(args.from_csv, args.zone, args.activity, args.label, bank)
    else:
        n = enroll_live(args.zone, args.activity, args.label, args.duration, args.ports, bank)

    count_after = bank.count()

    print(f"\n{'=' * 60}")
    print(f"  Enrolled {n} fingerprints")
    print(f"  Zone: {args.zone}  Activity: {args.activity}")
    print(f"  Collection: {count_before} -> {count_after} fingerprints")
    print(f"  All zones: {bank.get_all_zones()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()