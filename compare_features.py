"""
compare_features.py — Compare two feature-extractor CSV outputs side by side.

Reads two CSVs produced by feature_extractor_test.py, generates comparison
graphs showing differences in all 6 features.

Usage:
    python compare_features.py test1.csv test2.csv
    python compare_features.py test1.csv test2.csv --label1 "Empty room" --label2 "Person walking"
    python compare_features.py test1.csv test2.csv -o comparison.png
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURES = [
    "breathing_frequency",
    "total_energy",
    "doppler_mean",
    "variance_rx1",
    "variance_rx2",
    "variance_rx3",
]

FEATURE_UNITS = {
    "breathing_frequency": "Hz",
    "total_energy": "power",
    "doppler_mean": "Hz",
    "variance_rx1": "amplitude\u00b2",
    "variance_rx2": "amplitude\u00b2",
    "variance_rx3": "amplitude\u00b2",
}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: {path} is missing columns: {missing}")
        sys.exit(1)
    return df


def print_summary(df1, df2, label1, label2):
    """Print a summary statistics table for both tests."""
    print(f"\n{'Feature':<25s}  {'':^30s}  {'':^30s}")
    print(f"{'':25s}  {label1:^30s}  {label2:^30s}")
    print(f"{'':25s}  {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}"
          f"  {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    print("-" * 95)

    for feat in FEATURES:
        s1 = df1[feat]
        s2 = df2[feat]
        print(f"{feat:<25s}  "
              f"{s1.mean():8.3f} {s1.std():8.3f} {s1.min():8.3f} {s1.max():8.3f}  "
              f"{s2.mean():8.3f} {s2.std():8.3f} {s2.min():8.3f} {s2.max():8.3f}")
    print()


def plot_timeseries(df1, df2, label1, label2):
    """Overlay time-series for each feature."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Time Series: {label1} vs {label2}", fontsize=14, fontweight="bold")

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx // 2, idx % 2]
        ax.plot(df1["time_sec"], df1[feat], label=label1, alpha=0.8, linewidth=1.2)
        ax.plot(df2["time_sec"], df2[feat], label=label2, alpha=0.8, linewidth=1.2)
        ax.set_title(feat)
        ax.set_ylabel(FEATURE_UNITS[feat])
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_distributions(df1, df2, label1, label2):
    """Box plots comparing feature distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Distributions: {label1} vs {label2}", fontsize=14, fontweight="bold")

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx // 3, idx % 3]
        data = [df1[feat].values, df2[feat].values]
        bp = ax.boxplot(data, labels=[label1, label2], patch_artist=True,
                        widths=0.5)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(feat)
        ax.set_ylabel(FEATURE_UNITS[feat])
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_bar_means(df1, df2, label1, label2):
    """Bar chart of mean values with std error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Mean \u00b1 Std: {label1} vs {label2}", fontsize=14, fontweight="bold")

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx // 3, idx % 3]
        means = [df1[feat].mean(), df2[feat].mean()]
        stds = [df1[feat].std(), df2[feat].std()]
        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, width=0.5, capsize=5,
                      color=["#4C72B0", "#DD8452"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([label1, label2])
        ax.set_title(feat)
        ax.set_ylabel(FEATURE_UNITS[feat])
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bars with values.
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def plot_receiver_variance_radar(df1, df2, label1, label2):
    """Radar chart comparing mean receiver variances."""
    rx_features = ["variance_rx1", "variance_rx2", "variance_rx3"]
    labels = ["Rx1", "Rx2", "Rx3"]

    vals1 = [df1[f].mean() for f in rx_features]
    vals2 = [df2[f].mean() for f in rx_features]

    # Close the polygon.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals1 += vals1[:1]
    vals2 += vals2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals1, "o-", label=label1, linewidth=2)
    ax.fill(angles, vals1, alpha=0.15)
    ax.plot(angles, vals2, "o-", label=label2, linewidth=2)
    ax.fill(angles, vals2, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Receiver Variance (mean)", pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    return fig


def main():
    label1 = "Test 1"
    label2 = "Test 2"

    df1 = load_csv("test1.csv")
    df2 = load_csv("test2.csv")

    print(f"Loaded {label1}: {len(df1)} rows from test1.csv")
    print(f"Loaded {label2}: {len(df2)} rows from test2.csv")

    print_summary(df1, df2, label1, label2)

    fig1 = plot_timeseries(df1, df2, label1, label2)
    fig2 = plot_distributions(df1, df2, label1, label2)
    fig3 = plot_bar_means(df1, df2, label1, label2)
    fig4 = plot_receiver_variance_radar(df1, df2, label1, label2)

    plt.show()


if __name__ == "__main__":
    main()