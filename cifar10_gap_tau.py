"""Figure 4: four selected nonnegative CIFAR-10 LeNet gaps."""

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
    (1, 1, "red"),
    (2, 2, "orange"),
    (5, 0, "green"),
    (10, 0, "cyan"),
]


def run_tag(local_epochs, seed):
    if seed == 0 and local_epochs == 5:
        return "lenett500s0p9c0p2"
    if seed == 0:
        return "lenett500s0p9c0p2tau{}".format(local_epochs)
    return "lenett500s0p9c0p2tau{}seed{}".format(local_epochs, seed)


def result_paths(local_epochs, seed):
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma0p9_C0p2_"
        "clientdp_sens1_T500_lr0p1_tau{}_bs64_constant_seed{}_{}"
    ).format(local_epochs, seed, run_tag(local_epochs, seed))
    return (
        RESULT_DIR / ("loss_test_" + stem + ".npy"),
        RESULT_DIR / ("loss_train_" + stem + ".npy"),
    )


def load_gap(test_path, train_path, smooth=SMOOTH):
    test_loss = np.load(test_path, allow_pickle=False).astype(float)
    train_loss = np.load(train_path, allow_pickle=False).astype(float)
    length = min(len(test_loss), len(train_loss))
    # Paper convention: clamp each curve before smoothing.
    gap = np.maximum(test_loss[:length] - train_loss[:length], 0.0)
    if smooth > 1:
        gap = np.convolve(gap, np.ones(smooth) / smooth, mode="valid")
    return np.arange(len(gap)), gap


fig, ax = plt.subplots(figsize=[5, 4])

for local_epochs, seed, color in CURVES:
    test_path, train_path = result_paths(local_epochs, seed)
    if not test_path.exists() or not train_path.exists():
        raise FileNotFoundError(
            "Missing tau={}, seed={} loss result.".format(local_epochs, seed)
        )
    rounds, gap = load_gap(test_path, train_path)
    ax.plot(
        rounds,
        gap,
        label=r"$\tau={}$".format(local_epochs),
        color=color,
        linewidth=1.8,
    )

if not ax.lines:
    raise FileNotFoundError("No completed local-epoch result curves found.")

ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0.0, 0.18])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0.0, 0.181, 0.03))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Loss - Train Loss")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3)

fig.tight_layout()
plt.show()
