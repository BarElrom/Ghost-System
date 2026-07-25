import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("cleaned_csi.csv", header=None)

TIME_COL = 23
CSI_COL = 25
N_SUB = 64  # subcarriers

def parse_csi(csi_str):
    if pd.isna(csi_str):
        return None

    # robust: pull only integers (handles spaces, weird chars, trailing commas, etc.)
    nums = np.array([int(x) for x in re.findall(r"-?\d+", str(csi_str))], dtype=np.float32)

    # need at least 2*N_SUB values for I/Q pairs
    if nums.size < 2 * N_SUB:
        return None

    nums = nums[:2 * N_SUB]  # keep first 128 values
    I = nums[0::2]
    Q = nums[1::2]
    return np.abs(I + 1j * Q)

amps_list = df[CSI_COL].apply(parse_csi)
valid = amps_list.notna()

amplitudes = np.vstack(amps_list[valid].values)
time = df.loc[valid, TIME_COL].astype(float).values

# build axes
T, S = np.meshgrid(time, np.arange(N_SUB))

# 3D plot (with colors + colorbar)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    T, S, amplitudes.T,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)

fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Amplitude")

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Subcarrier Index")
ax.set_zlabel("Amplitude")

plt.tight_layout()
plt.show()

