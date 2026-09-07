"""v2 injector — Service 1: replay dataset CSI into the ESP32 receivers.

Loads a dataset via a matching adapter, normalizes each frame to the pipeline
contract (64 subcarriers, complex I/Q, 50 Hz), emits an empty-room calibration
preamble, then streams the recording as paced UDP frames.
"""
