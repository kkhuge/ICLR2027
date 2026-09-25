"""Plot CIFAR-10 LeNet accuracy for the 500-round sigma sweep at C=0.2."""

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


def result_path(sigma):
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma{}_C0p2_"
        "clientdp_sens1_T500_lr0p1_tau5_bs64_constant_seed0_{}"
    ).format(value_tag(sigma), run_tag(sigma))
    return RESULT_DIR / ("acc_test_" + stem + ".npy")


def load_curve(path, smooth=SMOOTH):
    accuracy = np.load(path, allow_pickle=True).astype(float)
    if smooth > 1:
        accuracy = np.convolve(
            accuracy, np.ones(smooth) / smooth, mode="valid",
        )
    return np.arange(len(accuracy)), accuracy


def curve_label(sigma):
    return r"$\sigma={:g}$".format(sigma)


fig, ax = plt.subplots(figsize=[5, 4])

for sigma, color, linestyle in CURVES:
    rounds, accuracy = load_curve(result_path(sigma))
    ax.plot(
        rounds,
        accuracy,
        label=curve_label(sigma),
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
    )

ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0.1, 0.5])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0.1, 0.51, 0.1))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3, loc="lower right")

fig.tight_layout()
plt.show()
