"""Plot test accuracy for the client-level CIFAR-10 LeNet C sweep."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


apply_plot_style()

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_acc" / "fedavg5"
NUM_ROUNDS = 500
SMOOTH = 10

CURVES = [
    (0.3, "tab:red"),
    (0.4, "tab:orange"),
    (0.5, "tab:green"),
]


def value_tag(value):
    return "{:g}".format(value).replace(".", "p")


def result_path(clip_norm):
    tag = value_tag(clip_norm)
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma0p9_C{}_"
        "clientdp_sens1_T500_lr0p1_tau5_bs64_constant_seed0_"
        "lenett500s0p9c{}"
    ).format(tag, tag)
    return RESULT_DIR / ("acc_test_" + stem + ".npy")


def load_curve(path, smooth=SMOOTH):
    accuracy = np.load(path, allow_pickle=True).astype(float)
    if smooth > 1:
        accuracy = np.convolve(
            accuracy, np.ones(smooth) / smooth, mode="valid",
        )
    return np.arange(len(accuracy)), accuracy


fig, ax = plt.subplots(figsize=[5, 4])

for clip_norm, color in CURVES:
    rounds, accuracy = load_curve(result_path(clip_norm))
    ax.plot(
        rounds,
        accuracy,
        label=r"$C={:g}$".format(clip_norm),
        color=color,
        linewidth=1.8,
    )

ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0.05, 0.50])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0.1, 0.51, 0.1))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3)

fig.tight_layout()
plt.show()
