"""Dataset adapters — normalize each dataset into the common injector contract.

Every adapter converts its native format into a stream of Snapshots
(64 complex subcarriers per available receiver stream). The injector handles
pacing, node fan-out, and the calibration preamble.
"""
