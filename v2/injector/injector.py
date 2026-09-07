

import logging
import time

import numpy as np

from v2.config_v2 import (
    CALIBRATION_SAMPLES,
    NODE_IDS,
    NUM_RECEIVERS,
    SAMPLE_RATE_HZ,
)
from v2.injector.adapters.base import DatasetAdapter
from v2.transport.udp_sender import UDPSender

logger = logging.getLogger("ghost.v2.injector")

_NODE_NAMES = sorted(NODE_IDS, key=NODE_IDS.get)

_DEFAULT_GAINS = {
    "RX1": complex(1.0, 0.0),
    "RX2": 0.9 * np.exp(1j * 0.10),
    "RX3": 0.8 * np.exp(1j * 0.20),
}


class Injector:
    """Replays a dataset adapter to the ESP32 nodes over UDP.

    Args:
        adapter: a DatasetAdapter.
        net_map: node -> (ip, port) override (defaults to config NODE_NET_MAP).
        sender: inject a UDPSender (else one is created from net_map).
        rate_hz: pacing rate; <= 0 sends as fast as possible (tests).
        calibration_samples: number of preamble frames per node.
        calib_window: number of head snapshots averaged into the baseline
            (defaults to min(calibration_samples, available)).
        gains: per-node complex gains for single-link fan-out.
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        net_map: dict | None = None,
        sender: UDPSender | None = None,
        rate_hz: float = float(SAMPLE_RATE_HZ),
        calibration_samples: int = CALIBRATION_SAMPLES,
        calib_window: int | None = None,
        gains: dict | None = None,
    ):
        self.adapter = adapter
        self._sender = sender if sender is not None else UDPSender(net_map=net_map)
        self._own_sender = sender is None
        self.rate_hz = rate_hz
        self.calibration_samples = calibration_samples
        self.calib_window = calib_window
        self.gains = gains or _DEFAULT_GAINS
        self.synthesized_nodes = adapter.num_streams < NUM_RECEIVERS
        self.stats = {name: {"calibration": 0, "operational": 0} for name in _NODE_NAMES}

    def _split_to_nodes(self, snapshot) -> dict:
        """Turn one snapshot into {node_name: complex64 (64,)} for all three nodes.

        Multi-stream datasets map stream i -> RX(i+1). Single-link datasets are
        copied to every node with a distinct complex gain (synthesized diversity).
        """
        if self.adapter.num_streams >= NUM_RECEIVERS:
            return {
                _NODE_NAMES[i]: np.asarray(snapshot.iq_by_stream[i], dtype=np.complex64)
                for i in range(NUM_RECEIVERS)
            }
        single = np.asarray(snapshot.iq_by_stream[0], dtype=np.complex64)
        return {name: (single * self.gains[name]).astype(np.complex64) for name in _NODE_NAMES}

    def _baseline_per_node(self, snapshots: list) -> dict:
        """Average the head snapshots into one empty-room baseline per node."""
        window = self.calib_window or min(self.calibration_samples, len(snapshots))
        window = max(1, min(window, len(snapshots)))
        head = [self._split_to_nodes(s) for s in snapshots[:window]]
        baseline = {
            name: np.mean([h[name] for h in head], axis=0).astype(np.complex64)
            for name in _NODE_NAMES
        }
        if logger.isEnabledFor(logging.INFO):
            for name in _NODE_NAMES:
                logger.info("baseline %s: |H_static| mean=%.3f over %d head snapshots",
                            name, float(np.abs(baseline[name]).mean()), window)
        return baseline

    def _send_tick(self, seq: dict, iq_per_node: dict, is_calibration: bool) -> None:
        """Send one frame to each node and advance its sequence + stat counter."""
        kind = "calibration" if is_calibration else "operational"
        for name in _NODE_NAMES:
            self._sender.send(name, seq[name], iq_per_node[name], calibration=is_calibration)
            seq[name] += 1
            self.stats[name][kind] += 1
        if self.rate_hz and self.rate_hz > 0:
            time.sleep(1.0 / self.rate_hz)

    def run(self, max_frames: int | None = None) -> dict:
        """Send the calibration preamble, then the operational stream. Returns stats."""
        snapshots = list(self.adapter.snapshots())
        if max_frames is not None:
            snapshots = snapshots[:max_frames]
        if not snapshots:
            logger.warning("adapter produced no snapshots; nothing to inject")
            return self.stats

        if self.synthesized_nodes:
            logger.warning(
                "Dataset '%s' has %d stream(s) < %d receivers; synthesizing node "
                "diversity — POSITION OUTPUT IS NON-METRIC (Plan 10.1).",
                self.adapter.name, self.adapter.num_streams, NUM_RECEIVERS,
            )
            gains_str = ", ".join(f"{n}=|{abs(g):.2f}|∠{np.angle(g):+.2f}rad"
                                  for n, g in self.gains.items())
            logger.info("fan-out gains (single-link → 3 nodes): %s", gains_str)
        else:
            logger.info("fan-out: %d dataset streams mapped directly to RX1..RX%d (metric)",
                        self.adapter.num_streams, NUM_RECEIVERS)

        baseline = self._baseline_per_node(snapshots)
        seq = {name: 0 for name in _NODE_NAMES}

        for _ in range(self.calibration_samples):
            self._send_tick(seq, baseline, is_calibration=True)
        for snapshot in snapshots:
            self._send_tick(seq, self._split_to_nodes(snapshot), is_calibration=False)

        logger.info("Injection complete: %d calib + %d operational frames per node",
                    self.calibration_samples, len(snapshots))
        return self.stats

    def close(self) -> None:
        """Close the UDP sender if this injector created it."""
        if self._own_sender:
            self._sender.close()


def build_adapter(dataset: str, path: str, **kwargs) -> DatasetAdapter:
    """Instantiate the adapter for a dataset name."""
    from v2.injector.adapters.embedded_wifi import EmbeddedWiFiAdapter
    from v2.injector.adapters.csi_bench import CSIBenchAdapter
    from v2.injector.adapters.intel_resp import IntelRespAdapter

    registry = {
        "embedded_wifi": EmbeddedWiFiAdapter,
        "csi_bench": CSIBenchAdapter,
        "intel_resp": IntelRespAdapter,
    }
    if dataset not in registry:
        raise ValueError(f"unknown dataset {dataset!r}; known: {list(registry)}")
    return registry[dataset](path, **kwargs)


def _main() -> None:
    import argparse

    from v2.config_v2 import NODE_NET_MAP

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    ap = argparse.ArgumentParser(description="GHOST v2 injector — replay a dataset into the ESP32 nodes over UDP")
    ap.add_argument("--dataset", default="embedded_wifi", help="dataset adapter name")
    ap.add_argument("--path", required=True, help="dataset file path")
    ap.add_argument("--rate", type=float, default=float(SAMPLE_RATE_HZ), help="pacing rate Hz (0 = as fast as possible)")
    ap.add_argument("--calib", type=int, default=CALIBRATION_SAMPLES, help="calibration preamble frames")
    ap.add_argument("--calib-window", type=int, default=None, help="head snapshots averaged into baseline")
    ap.add_argument("--max-frames", type=int, default=None, help="cap operational frames")
    ap.add_argument("--host", default=None, help="send all nodes to this IP (else config NODE_NET_MAP)")
    ap.add_argument("--port", type=int, default=None, help="port for --host")
    args = ap.parse_args()

    net_map = None
    if args.host is not None:
        port = args.port if args.port is not None else next(iter(NODE_NET_MAP.values()))[1]
        net_map = {name: (args.host, port) for name in _NODE_NAMES}

    adapter = build_adapter(args.dataset, args.path)
    inj = Injector(
        adapter, net_map=net_map, rate_hz=args.rate,
        calibration_samples=args.calib, calib_window=args.calib_window,
    )
    try:
        stats = inj.run(max_frames=args.max_frames)
        print("Injection stats:", stats)
    finally:
        inj.close()


if __name__ == "__main__":
    _main()
