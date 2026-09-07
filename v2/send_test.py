"""
Hardware bring-up helper: send a few synthetic frames to one ESP32 node.

Watch the target board's serial (`screen /dev/cu.usbserial-XXXX 115200`) and
you should see one CSI_DATA line per frame — proving the Wi-Fi -> ESP32 -> USB
loopback works end to end.

Usage (run from the project root):
    python v2/send_test.py 192.168.1.51                 # 10 frames to RX1
    python v2/send_test.py 192.168.1.52 --node RX2      # to RX2
    python v2/send_test.py 192.168.1.51 --count 50      # more frames
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from v2.transport.udp_sender import UDPSender


def main() -> None:
    ap = argparse.ArgumentParser(description="Send test frames to one ESP32 node")
    ap.add_argument("ip", help="target board IP")
    ap.add_argument("--node", default="RX1", help="node name: RX1 / RX2 / RX3")
    ap.add_argument("--port", type=int, default=5005, help="UDP port")
    ap.add_argument("--count", type=int, default=10, help="number of frames")
    args = ap.parse_args()

    sender = UDPSender(net_map={args.node: (args.ip, args.port)})
    for i in range(args.count):
        iq = (np.arange(64) + 1j * np.arange(64)).astype("complex64")
        sender.send(args.node, i, iq)
        time.sleep(0.1)
    sender.close()
    print(f"sent {args.count} frames to {args.node} @ {args.ip}:{args.port}")


if __name__ == "__main__":
    main()
