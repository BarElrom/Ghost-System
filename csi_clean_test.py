"""
csi_clean_test.py — Read raw CSI CSV, run the Signal Cleaner, write cleaned CSV.

Input CSV format (no header, produced by gateway_csv_test.py):
    Columns 0-22 : ESP32 metadata
    Column  23   : elapsed time (float seconds)
    Column  24   : first_word
    Column  25   : raw CSI I/Q array string, e.g. "[1,2,3,...,128]"

Output CSV format (no header):
    Columns 0-22 : metadata (passed through)
    Column  23   : elapsed time (passed through)
    Column  24   : first_word (passed through)
    Column  25   : cleaned amplitude array, e.g. "[0.12,0.45,...,0.78]"

Usage:
    python csi_clean_test.py                                # defaults
    python csi_clean_test.py -i raw.csv -o cleaned.csv
    python csi_clean_test.py --fs 100 --lowcut 0.1 --highcut 4.0
"""

import csv
import re
import argparse

import numpy as np

from signal_cleaner import SignalCleaner

N_SUB = 64
DEFAULT_INPUT = "example_csi.csv"
DEFAULT_OUTPUT = "cleaned_csi.csv"

TIME_COL = 23
CSI_COL = 25


def parse_amplitude(csi_str: str) -> np.ndarray | None:
    """Parse I/Q string into amplitude array of shape (64,)."""
    nums = np.array(
        [int(x) for x in re.findall(r"-?\d+", str(csi_str))],
        dtype=np.float32,
    )
    if nums.size < 2 * N_SUB:
        return None
    nums = nums[: 2 * N_SUB]
    i_vals = nums[0::2]
    q_vals = nums[1::2]
    return np.sqrt(i_vals**2 + q_vals**2)


def main():
    ap = argparse.ArgumentParser(
        description="Clean raw CSI data and write cleaned CSV"
    )
    ap.add_argument("-i", "--input", default=DEFAULT_INPUT,
                    help=f"Input CSV path (default: {DEFAULT_INPUT})")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                    help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--fs", type=float, default=100.0,
                    help="Sampling frequency in Hz (default: 100)")
    ap.add_argument("--lowcut", type=float, default=0.1,
                    help="Bandpass low cutoff Hz (default: 0.1)")
    ap.add_argument("--highcut", type=float, default=4.0,
                    help="Bandpass high cutoff Hz (default: 4.0)")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Read CSV and parse amplitudes
    # ------------------------------------------------------------------
    rows = []
    amplitudes = []

    with open(args.input, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= CSI_COL:
                continue
            amp = parse_amplitude(row[CSI_COL])
            if amp is None:
                continue
            rows.append(row)
            amplitudes.append(amp)

    if not amplitudes:
        print(f"No valid CSI rows found in {args.input}")
        return

    n_packets = len(amplitudes)
    raw_matrix = np.stack(amplitudes, axis=1)  # (64, T)
    raw_matrix = raw_matrix[np.newaxis, :, :]  # (1, 64, T) — single receiver

    print(f"Read {n_packets} packets from {args.input}")
    print(f"Raw matrix shape: {raw_matrix.shape}")
    print(f"Raw amplitude — mean: {raw_matrix.mean():.2f}  std: {raw_matrix.std():.2f}")

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    cleaner = SignalCleaner(
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
    )
    cleaned_matrix = cleaner.clean(raw_matrix)  # (1, 64, T)

    print(f"Cleaned amplitude — mean: {cleaned_matrix.mean():.4f}  std: {cleaned_matrix.std():.4f}")

    # ------------------------------------------------------------------
    # 3. Write cleaned CSV
    # ------------------------------------------------------------------
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        for t in range(n_packets):
            original_row = rows[t]
            cleaned_amps = cleaned_matrix[0, :, t]  # (64,)

            # Format cleaned amplitudes as a bracket-delimited array string.
            amp_str = "[" + ",".join(f"{v:.4f}" for v in cleaned_amps) + "]"

            # Preserve metadata (cols 0-22), time (23), first_word (24),
            # replace CSI col (25) with cleaned amplitudes.
            out_row = original_row[:CSI_COL] + [amp_str]
            writer.writerow(out_row)

    print(f"\nCleaned CSV written to {args.output}  ({n_packets} rows)")
    print(f"Visualize with:  python csiTest.py  (point it at {args.output})")


if __name__ == "__main__":
    main()