"""Figure 4: four selected CIFAR-10 LeNet local-epoch curves."""

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


def result_path(local_epochs, seed):
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma0p9_C0p2_"
        "clientdp_sens1_T500_lr0p1_tau{}_bs64_constant_seed{}_{}"
    ).format(local_epochs, seed, run_tag(local_epochs, seed))
    return RESULT_DIR / ("acc_test_" + stem + ".npy")


def load_curve(path, smooth=SMOOTH):
    accuracy = np.load(path, allow_pickle=False).astype(float)
    if smooth > 1:
        accuracy = np.convolve(
            accuracy, np.ones(smooth) / smooth, mode="valid",
        )
    return np.arange(len(accuracy)), accuracy


fig, ax = plt.subplots(figsize=[5, 4])

for local_epochs, seed, color in CURVES:
    path = result_path(local_epochs, seed)
    if not path.exists():
        raise FileNotFoundError(
            "Missing tau={}, seed={} result: {}".format(
                local_epochs, seed, path
            )
        )
    rounds, accuracy = load_curve(path)
    ax.plot(
        rounds,
        accuracy,
        label=r"$\tau={}$".format(local_epochs),
        color=color,
        linewidth=1.8,
    )

if not ax.lines:
    raise FileNotFoundError("No completed local-epoch result curves found.")

ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0, 0.50])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3)

fig.tight_layout()
plt.show()
