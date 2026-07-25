"""
memory_bank_test.py — Integration test for the MemoryBank.

Requires a running ChromaDB server (docker compose up -d).
Tests enrollment, querying, batch operations, zone filtering, and
edge cases against a dedicated test collection.

Usage:
    docker compose up -d
    python memory_bank_test.py
"""

import sys
import logging

import numpy as np

from config import ChromaSettings
from memory_bank import MemoryBank
from feature_extractor import NUM_FEATURES

# Use a separate collection so tests don't pollute real data.
TEST_SETTINGS = ChromaSettings(collection_name="ghost_test_collection")

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def make_vector(*values) -> np.ndarray:
    """Create a (6,) state vector from positional args."""
    return np.array(values, dtype=np.float64)


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    print("=" * 60)
    print("  GHOST MemoryBank — Integration Tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Health check
    # ------------------------------------------------------------------
    print("\n[1] Health Check")
    try:
        bank = MemoryBank(settings=TEST_SETTINGS)
        ok = bank.health_check()
        report("ChromaDB reachable", ok)
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to ChromaDB — {e}")
        print("  Make sure the container is running: docker compose up -d")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 2: Clear collection (start clean)
    # ------------------------------------------------------------------
    print("\n[2] Clear Collection")
    bank.clear()
    report("Collection cleared", bank.count() == 0, f"count={bank.count()}")

    # ------------------------------------------------------------------
    # Test 3: Single enrollment
    # ------------------------------------------------------------------
    print("\n[3] Single Enrollment")
    kitchen_vec = make_vector(0.30, 1500.0, 0.80, 0.012, 0.008, 0.015)
    fid = bank.enroll(kitchen_vec, zone="Kitchen", activity="Static")
    report("Enroll returns ID", isinstance(fid, str) and len(fid) > 0)
    report("Count is 1", bank.count() == 1, f"count={bank.count()}")

    # ------------------------------------------------------------------
    # Test 4: Query — exact match
    # ------------------------------------------------------------------
    print("\n[4] Query — Exact Match")
    candidates = bank.query(kitchen_vec, n_results=1)
    report("Returns 1 candidate", len(candidates) == 1)
    if candidates:
        c = candidates[0]
        report("Zone is Kitchen", c.zone == "Kitchen", f"got '{c.zone}'")
        report("Activity is Static", c.activity == "Static", f"got '{c.activity}'")
        report("Score ~1.0 (exact match)", c.score > 0.99, f"score={c.score}")

    # ------------------------------------------------------------------
    # Test 5: Enroll more zones
    # ------------------------------------------------------------------
    print("\n[5] Enroll Additional Zones")
    bedroom_vec = make_vector(0.25, 800.0, 0.40, 0.005, 0.020, 0.003)
    hall_vec = make_vector(2.50, 2200.0, 3.80, 0.080, 0.070, 0.090)
    empty_vec = make_vector(0.00, 10.0, 0.01, 0.001, 0.001, 0.001)

    bank.enroll(bedroom_vec, zone="Bedroom", activity="Breathing")
    bank.enroll(hall_vec, zone="Hall", activity="Walking")
    bank.enroll(empty_vec, zone="Empty", activity="None")
    report("Count is 4", bank.count() == 4, f"count={bank.count()}")

    # ------------------------------------------------------------------
    # Test 6: Query — nearest match (Kitchen-like vector)
    # ------------------------------------------------------------------
    print("\n[6] Query — Nearest Match")
    query_vec = make_vector(0.31, 1480.0, 0.82, 0.011, 0.009, 0.014)
    candidates = bank.query(query_vec, n_results=3)
    report("Returns 3 candidates", len(candidates) == 3)
    if candidates:
        report(
            "Top candidate is Kitchen",
            candidates[0].zone == "Kitchen",
            f"got '{candidates[0].zone}' (score={candidates[0].score})",
        )

    # ------------------------------------------------------------------
    # Test 7: Query — Hall-like vector
    # ------------------------------------------------------------------
    print("\n[7] Query — Hall-like Vector")
    hall_query = make_vector(2.40, 2100.0, 3.70, 0.075, 0.065, 0.085)
    candidates = bank.query(hall_query, n_results=1)
    if candidates:
        report(
            "Top candidate is Hall",
            candidates[0].zone == "Hall",
            f"got '{candidates[0].zone}' (score={candidates[0].score})",
        )

    # ------------------------------------------------------------------
    # Test 8: Zone filter
    # ------------------------------------------------------------------
    print("\n[8] Zone Filter")
    candidates = bank.query(kitchen_vec, n_results=3, zone_filter="Bedroom")
    report("Returns results", len(candidates) >= 1)
    if candidates:
        report(
            "All results are Bedroom",
            all(c.zone == "Bedroom" for c in candidates),
            f"zones={[c.zone for c in candidates]}",
        )

    # ------------------------------------------------------------------
    # Test 9: Batch enrollment
    # ------------------------------------------------------------------
    print("\n[9] Batch Enrollment")
    batch_vecs = [
        make_vector(0.32, 1520.0, 0.83, 0.013, 0.007, 0.016),
        make_vector(0.29, 1470.0, 0.78, 0.011, 0.009, 0.013),
        make_vector(0.33, 1550.0, 0.85, 0.014, 0.006, 0.017),
    ]
    batch_zones = ["Kitchen", "Kitchen", "Kitchen"]
    batch_activities = ["Static", "Static", "Static"]

    ids = bank.enroll_batch(batch_vecs, batch_zones, batch_activities)
    report("Batch returns 3 IDs", len(ids) == 3)
    report("Count is 7", bank.count() == 7, f"count={bank.count()}")

    # ------------------------------------------------------------------
    # Test 10: get_all_zones
    # ------------------------------------------------------------------
    print("\n[10] Get All Zones")
    zones = bank.get_all_zones()
    expected = ["Bedroom", "Empty", "Hall", "Kitchen"]
    report("Zones match", zones == expected, f"got {zones}")

    # ------------------------------------------------------------------
    # Test 11: Validation — wrong shape
    # ------------------------------------------------------------------
    print("\n[11] Validation — Wrong Shape")
    try:
        bad_vec = np.array([1.0, 2.0, 3.0])
        bank.enroll(bad_vec, zone="Bad", activity="Bad")
        report("Rejects wrong shape", False, "no exception raised")
    except ValueError:
        report("Rejects wrong shape", True)

    # ------------------------------------------------------------------
    # Test 12: Validation — wrong type
    # ------------------------------------------------------------------
    print("\n[12] Validation — Wrong Type")
    try:
        bank.enroll([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], zone="Bad", activity="Bad")
        report("Rejects non-ndarray", False, "no exception raised")
    except TypeError:
        report("Rejects non-ndarray", True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n[Cleanup]")
    bank.clear()
    report("Collection cleared", bank.count() == 0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()