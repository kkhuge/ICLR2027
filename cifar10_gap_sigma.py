"""Plot the nonnegative CIFAR-10 LeNet gap for the C=0.2 sigma sweep."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


apply_plot_style()

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_loss" / "fedavg5"
NUM_ROUNDS = 500
SMOOTH = 10

CURVES = [
    (0.6, "tab:red", "-"),
    (0.9, "tab:orange", "-"),
    (1.2, "tab:green", "-"),
    (1.5, "tab:cyan", "-"),
]


def value_tag(value):
    return "{:g}".format(value).replace(".", "p")


def run_tag(sigma):
    # sigma=0.9 reuses the completed C-sweep result.
    if np.isclose(sigma, 0.9):
        return "lenett500s0p9c0p2"
    return "lenett500c0p2s{}".format(value_tag(sigma))


def result_paths(sigma):
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma{}_C0p2_"
        "clientdp_sens1_T500_lr0p1_tau5_bs64_constant_seed0_{}"
    ).format(value_tag(sigma), run_tag(sigma))
    return (
        RESULT_DIR / ("loss_test_" + stem + ".npy"),
        RESULT_DIR / ("loss_train_" + stem + ".npy"),
    )


def load_gap(test_path, train_path, smooth=SMOOTH):
    test_loss = np.load(test_path, allow_pickle=True).astype(float)
    train_loss = np.load(train_path, allow_pickle=True).astype(float)
    length = min(len(test_loss), len(train_loss))
    # Paper convention: clip negative gaps before smoothing.
    gap = np.maximum(test_loss[:length] - train_loss[:length], 0.0)
    if smooth > 1:
        gap = np.convolve(gap, np.ones(smooth) / smooth, mode="valid")
    return np.arange(len(gap)), gap


def curve_label(sigma):
    return r"$\sigma={:g}$".format(sigma)


fig, ax = plt.subplots(figsize=[5, 4])

for sigma, color, linestyle in CURVES:
    test_path, train_path = result_paths(sigma)
    rounds, gap = load_gap(test_path, train_path)
    ax.plot(
        rounds,
        gap,
        label=curve_label(sigma),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )

ax.axhline(y=0, linestyle="--", color="black", linewidth=1)
ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0.0, 0.20])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0.0, 0.201, 0.05))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Loss - Train Loss")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3)

fig.tight_layout()
plt.show()
