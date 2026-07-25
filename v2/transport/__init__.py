"""v2 transport layer — UDP wire frame codec, sender, and software ESP32 mock.

This package implements Plan section 2 (Physical & Transport architecture):
the exact UDP payload the injector sends into the ESP32 receivers, and a
software stand-in for the ESP32 firmware that re-emits CSI_DATA serial lines.
"""
