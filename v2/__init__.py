"""GHOST System v2 — hardware-injection (loopback) mode.

All v2 code is additive and isolated under this package. The existing root
pipeline (gateway.py, signal_cleaner.py, ...) is never modified; where v2
behavior must differ, the file is duplicated into v2/ and edited there.

See v2/GHOST_V2_PLAN.md for the full design.
"""