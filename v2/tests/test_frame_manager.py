"""Phase 0 tests — FrameManager cross-receiver alignment.

Proves the manager (1) joins receivers by sequential position / receive time so
that a matrix column is genuinely simultaneous across receivers, (2) drops ticks
where a receiver is missing (lost packets), (3) selects a sensible reference and
tolerates timestamp jitter, and (4) emits a linear (monotonic, zero-based) time
axis. Also covers the ComplexMatrix.get_streams snapshot the manager consumes.

Run:  python v2/tests/test_frame_manager.py
"""

import sys

from _harness import Harness

import numpy as np

from v2.config_v2 import NUM_SUBCARRIERS, SAMPLE_RATE_HZ
from v2.ghost.frame_manager import FrameManager
from v2.ghost.gateway_v2 import ComplexMatrix


def _tag_csi(rx: int, seq: int) -> np.ndarray:
    """A (64,) complex sample whose real part encodes (rx, seq) for verification."""
    return np.full(NUM_SUBCARRIERS, rx * 1000 + seq, dtype=np.complex64)


def _stream(rx: int, seqs: list[int], *, host_step_ns: int = 0,
            host_start_ns: int = 0, jitter_ns=None) -> dict:
    """Build a get_streams-shaped dict for one receiver from a list of seq ids."""
    seqs = list(seqs)
    csi = np.stack([_tag_csi(rx, s) for s in seqs], axis=1) if seqs \
        else np.zeros((NUM_SUBCARRIERS, 0), dtype=np.complex64)
    device_ts = np.array([s * (1_000_000 // int(SAMPLE_RATE_HZ)) for s in seqs], dtype=np.int64)
    if host_step_ns:
        host = np.array([host_start_ns + i * host_step_ns for i in range(len(seqs))],
                        dtype=np.int64)
    else:
        host = np.zeros(len(seqs), dtype=np.int64)
    if jitter_ns is not None:
        host = host + np.array(jitter_ns, dtype=np.int64)
    return {"csi": csi, "device_ts": device_ts, "host_ts": host,
            "seq": np.array(seqs, dtype=np.int64)}


def test_perfect_alignment(h: Harness) -> None:
    mgr = FrameManager(num_receivers=3, align_on="seq")
    streams = [_stream(rx, list(range(10))) for rx in range(3)]
    win = mgr.align(streams)
    h.expect("all 10 ticks aligned", win.num_frames == 10, str(win.num_frames))
    h.expect("nothing dropped", win.dropped == 0)
    h.expect("shape [3,64,10]", win.matrix.shape == (3, NUM_SUBCARRIERS, 10),
             str(win.matrix.shape))
    # Column t must be (rx, seq=t) for every receiver — proves correct gather.
    ok = all(win.matrix[rx, 0, t].real == rx * 1000 + t
             for rx in range(3) for t in range(10))
    h.expect("each column is cross-rx simultaneous (rx,seq) tags match", ok)
    h.expect("time is zero-based", win.time_s[0] == 0.0)
    dt = np.diff(win.time_s)
    h.expect("linear time step = 1/fs", np.allclose(dt, 1.0 / SAMPLE_RATE_HZ))
    h.expect("full coverage", all(v == 1.0 for v in win.coverage.values()))


def test_dropped_on_lost_packet(h: Harness) -> None:
    # rx1 lost seq 5; seq-join must drop that tick, not misalign the rest.
    mgr = FrameManager(num_receivers=3, align_on="seq")
    streams = [
        _stream(0, list(range(10))),
        _stream(1, [s for s in range(10) if s != 5]),
        _stream(2, list(range(10))),
    ]
    win = mgr.align(streams)
    h.expect("9 ticks survive", win.num_frames == 9, str(win.num_frames))
    h.expect("one tick dropped", win.dropped == 1, str(win.dropped))
    h.expect("seq 5 absent from aligned seqs", 5 not in set(win.seq.tolist()))
    # Decode seq from each tag (real = rx*1000 + seq): every receiver in a column
    # must sit on that column's seq — i.e. nobody slid across the rx1 gap.
    ok = all((win.matrix[rx, 0, t].real - rx * 1000) == win.seq[t]
             for rx in range(3) for t in range(win.num_frames))
    h.expect("every receiver stays on the column's seq after the gap", ok)


def test_ref_is_most_complete(h: Harness) -> None:
    # rx0 sparse, rx1 full → rx1 should be the reference; overlap is seqs 2,4,6,8.
    mgr = FrameManager(num_receivers=2, align_on="seq")
    streams = [_stream(0, [2, 4, 6, 8]), _stream(1, list(range(10)))]
    win = mgr.align(streams)
    h.expect("aligned to the sparse receiver's overlap", win.num_frames == 4,
             str(win.num_frames))
    h.expect("aligned seqs are 2,4,6,8", win.seq.tolist() == [2, 4, 6, 8],
             str(win.seq.tolist()))


def test_host_ts_within_tolerance(h: Harness) -> None:
    # Same seqs, but align on host clock; rx1 jittered by < half a frame period.
    fs = int(SAMPLE_RATE_HZ)
    step = 1_000_000_000 // fs  # ns per frame
    jitter = [int(0.3 * step)] * 10  # 30% of a period — inside the 50% tolerance
    mgr = FrameManager(num_receivers=2, align_on="host_ts", tolerance_ratio=0.5)
    streams = [
        _stream(0, list(range(10)), host_step_ns=step),
        _stream(1, list(range(10)), host_step_ns=step, jitter_ns=jitter),
    ]
    win = mgr.align(streams)
    h.expect("jitter under tolerance keeps all ticks", win.num_frames == 10,
             str(win.num_frames))
    ok = all(win.matrix[1, 0, t].real == 1000 + t for t in range(10))
    h.expect("host-ts nearest join picked the right rx1 samples", ok)


def test_host_ts_beyond_tolerance_drops(h: Harness) -> None:
    fs = int(SAMPLE_RATE_HZ)
    step = 1_000_000_000 // fs
    # rx1 shifted by a whole period → no host_ts falls within half-period tolerance.
    mgr = FrameManager(num_receivers=2, align_on="host_ts", tolerance_ratio=0.4)
    # A half-period constant offset: rx1's nearest sample is always 0.5·period
    # away — beyond the 0.4·period tolerance — so no tick matches. (A whole-period
    # offset would merely re-index and wrongly appear to align.)
    streams = [
        _stream(0, list(range(10)), host_step_ns=step),
        _stream(1, list(range(10)), host_step_ns=step, host_start_ns=step // 2),
    ]
    win = mgr.align(streams)
    h.expect("out-of-tolerance offset drops all ticks", win.num_frames == 0,
             str(win.num_frames))


def test_empty_stream(h: Harness) -> None:
    mgr = FrameManager(num_receivers=3, align_on="seq")
    streams = [_stream(0, list(range(5))), _stream(1, []), _stream(2, list(range(5)))]
    win = mgr.align(streams)
    h.expect("empty receiver → empty window", win.num_frames == 0)
    h.expect("empty window shape [3,64,0]", win.matrix.shape == (3, NUM_SUBCARRIERS, 0))


def test_wrong_stream_count_raises(h: Harness) -> None:
    mgr = FrameManager(num_receivers=3, align_on="seq")
    h.expect_raises("mismatched stream count raises", ValueError,
                    lambda: mgr.align([_stream(0, [0, 1])]))


def test_bad_align_on_raises(h: Harness) -> None:
    h.expect_raises("unknown align_on raises", ValueError,
                    lambda: FrameManager(align_on="wallclock"))


def test_integration_with_matrix(h: Harness) -> None:
    # End-to-end: append complex + timestamps to ComplexMatrix, snapshot streams,
    # align. Proves get_streams feeds the manager and timestamps survive.
    m = ComplexMatrix(max_time=64)
    for seq in range(8):
        for rx in range(3):
            m.append(rx, _tag_csi(rx, seq), device_ts=seq * 10_000,
                     host_ts=seq * 1_000_000, seq_id=seq)
    streams = m.get_streams(8)
    h.expect("get_streams returns 3 receivers", len(streams) == 3)
    h.expect("stream carries seq", streams[0]["seq"].tolist() == list(range(8)))
    h.expect("stream carries device_ts", streams[0]["device_ts"][1] == 10_000)
    h.expect("stream carries host_ts", streams[0]["host_ts"][2] == 2_000_000)

    win = FrameManager(num_receivers=3, align_on="seq").align(streams)
    h.expect("integration aligns 8 ticks", win.num_frames == 8, str(win.num_frames))
    ok = all(win.matrix[rx, 0, t].real == rx * 1000 + t
             for rx in range(3) for t in range(8))
    h.expect("integration tags line up", ok)


def main() -> int:
    h = Harness("frame_manager (cross-receiver alignment)")
    h.case("perfect_alignment", test_perfect_alignment)
    h.case("dropped_on_lost_packet", test_dropped_on_lost_packet)
    h.case("ref_is_most_complete", test_ref_is_most_complete)
    h.case("host_ts_within_tolerance", test_host_ts_within_tolerance)
    h.case("host_ts_beyond_tolerance_drops", test_host_ts_beyond_tolerance_drops)
    h.case("empty_stream", test_empty_stream)
    h.case("wrong_stream_count_raises", test_wrong_stream_count_raises)
    h.case("bad_align_on_raises", test_bad_align_on_raises)
    h.case("integration_with_matrix", test_integration_with_matrix)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
