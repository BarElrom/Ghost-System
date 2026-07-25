"""
GHOST v2 — Phase 2 end-to-end demo on a REAL dataset.

Runs the full hardware-free chain on an actual ESP32 CSI capture:

    dataset file
        -> Injector (calibration preamble + operational stream, UDP)
        -> UDPSource -> GatewayV2 (complex I/Q buffer)
        -> SignalCleanerV2 (static subtraction, calibrated from the preamble)
        -> FeatureExtractorV2 (motion features + per-node energies)

It prints a before/after amplitude comparison (proving the static baseline was
removed) and the extracted feature set.

Usage:
    python v2/run_phase2_demo.py                       # uses example_csi.csv
    python v2/run_phase2_demo.py --path my_capture.csv --max-frames 1000 --calib 200
"""

import argparse
import os
import sys
import threading
import time

# Repo root on path so `import v2...` works when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from v2.ghost.gateway_v2 import GatewayV2
from v2.ghost.sources import UDPSource
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2
from v2.ghost.feature_extractor_v2 import FeatureExtractorV2
from v2.injector.injector import Injector, build_adapter, _NODE_NAMES


def main() -> int:
    ap = argparse.ArgumentParser(description="GHOST v2 Phase 2 end-to-end demo")
    ap.add_argument("--path", default="example_csi.csv", help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--max-frames", type=int, default=800, help="operational frames to replay")
    ap.add_argument("--calib", type=int, default=200, help="calibration preamble frames")
    ap.add_argument("--rate", type=float, default=0.0, help="pacing Hz (0 = fast)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"dataset not found: {args.path}")
        return 1

    adapter = build_adapter(args.dataset, args.path)
    n_op = args.max_frames
    buffer_size = args.calib + n_op + 50

    udp = UDPSource(bind_host="127.0.0.1", bind_port=0)
    gw = GatewayV2(sources=[udp], buffer_size=buffer_size)
    gw.start()

    inj = Injector(
        adapter,
        net_map={n: ("127.0.0.1", udp.port) for n in _NODE_NAMES},
        rate_hz=args.rate,
        calibration_samples=args.calib,
    )

    print(f"Injecting '{args.path}' via '{args.dataset}' adapter "
          f"({args.calib} calib + up to {n_op} operational frames)...")

    injection_thread = threading.Thread(target=lambda: inj.run(max_frames=n_op), daemon=True)
    injection_thread.start()
    injection_thread.join()

    # Wait for the gateway to finish draining (received count stops growing).
    previous, stable_ticks = -1, 0
    for _ in range(200):
        time.sleep(0.03)
        received = min(gw.get_matrix().get_receiver_count(r) for r in range(3))
        stable_ticks = stable_ticks + 1 if received == previous else 0
        previous = received
        if stable_ticks >= 3 and received > 0:
            break

    counts = [gw.get_matrix().get_receiver_count(r) for r in range(3)]
    total = min(counts)
    # How many frames the injector actually sent per node (from its own stats).
    sent = inj.stats["RX1"]["calibration"] + inj.stats["RX1"]["operational"]
    print(f"\nFrames received per node: {counts}  (sent {sent})")
    expected = sent
    if total < expected:
        print(f"  note: {expected - total} frame(s) not received (UDP burst) — demo continues")
    if total <= args.calib:
        print("  ERROR: not enough frames to separate calibration from operational")
        gw.stop()
        return 1

    # Chronological window: [calibration preamble | operational].
    window = gw.get_matrix().get_latest(total)          # (3, 64, total)
    calib_slice = window[:, :, : args.calib]
    op_slice = window[:, :, args.calib:]

    cleaner = SignalCleanerV2()
    cleaner.calibrate_from(calib_slice)
    cleaned = cleaner.clean(op_slice)

    extractor = FeatureExtractorV2()
    feats = extractor.extract(cleaned)

    # --- report ---
    raw_amp = float(np.mean(np.abs(op_slice)))
    dyn_amp = float(np.mean(np.abs(cleaned.dynamic)))                 # post-subtraction, pre-bandpass
    motion_std = float(np.std(cleaned.amplitude))                     # bandpassed motion energy

    # Phase-coherence diagnostic measured on the RAW operational frames of RX1
    # (gain == 1, so op_slice[0] is the untouched dataset CSI). If the complex
    # mean is much smaller than the mean amplitude, packet phase is drifting and
    # complex static subtraction cannot form a clean baseline.
    rx0 = op_slice[0]                                    # (64, T) complex, raw
    meanabs = np.mean(np.abs(rx0), axis=1)
    absmean = np.abs(np.mean(rx0, axis=1))
    active = meanabs > 1e-6
    coherence = float(np.mean(absmean[active] / meanabs[active])) if active.any() else 0.0

    print("\n=== Signal cleaning (static subtraction) ===")
    print(f"  raw operational   |H_raw|      mean : {raw_amp:8.3f}")
    print(f"  after subtraction |H_dynamic|  mean : {dyn_amp:8.3f}")
    print(f"  bandpassed motion  A_clean     std  : {motion_std:8.3f}")
    print("\n  -- raw-CSI phase-coherence diagnostic (RX1) --")
    print(f"  per-frame amplitude mean |H|      : {float(meanabs[active].mean()):8.3f}")
    print(f"  complex mean |E[H]|               : {float(absmean[active].mean()):8.3f}")
    print(f"  coherence ratio (1=coherent)      : {coherence:8.3f}")
    if coherence < 0.5:
        print("  WARNING: low coherence — raw CSI has per-packet random phase, so the")
        print("           complex baseline partially cancels and static subtraction is")
        print("           weak. Needs phase sanitization (see Plan open items).")

    print("\n=== Extracted features ===")
    print(f"  breathing_frequency : {feats.breathing_frequency:.3f} Hz")
    print(f"  total_energy        : {feats.total_energy:.2f}")
    print(f"  doppler_mean        : {feats.doppler_mean:.3f} Hz")
    print(f"  variance_rx         : {[round(v, 3) for v in feats.variance_rx]}")
    print(f"  phase_variance_rx   : {[round(v, 3) for v in feats.phase_variance_rx]}")
    print(f"  node_energies       : {{"
          + ", ".join(f'{k}: {v:.2f}' for k, v in feats.node_energies.items()) + "}")

    gw.stop()
    print("\nPhase 2 end-to-end demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
