"""Figure 3: CIFAR-10 LeNet accuracy under different participation rates."""

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
    # (50, "tab:red"),
    (100, "tab:red"),
    (200, "tab:orange"),
    (500, "tab:green"),
]


def run_tag(clients_per_round):
    if clients_per_round == 50:
        return "lenett500s0p9c0p2"
    return "lenett500s0p9c0p2m{}".format(clients_per_round)


def result_path(clients_per_round):
    stem = (
        "cifar10_all_data_1_random_iid_lenet_sigma0p9_C0p2_"
        "clientdp_sens1_T500_lr0p1_tau5_bs64_constant_seed0_{}"
    ).format(run_tag(clients_per_round))
    return RESULT_DIR / ("acc_test_" + stem + ".npy")


def load_curve(path, smooth=SMOOTH):
    accuracy = np.load(path, allow_pickle=False).astype(float)
    if smooth > 1:
        accuracy = np.convolve(
            accuracy, np.ones(smooth) / smooth, mode="valid",
        )
    return np.arange(len(accuracy)), accuracy


fig, ax = plt.subplots(figsize=[5, 4])

for clients_per_round, color in CURVES:
    path = result_path(clients_per_round)
    if not path.exists():
        print("Skipping m={}: result not available yet".format(clients_per_round))
        continue
    rounds, accuracy = load_curve(path)
    ax.plot(
        rounds,
        accuracy,
        label=r"$m={}$".format(clients_per_round),
        color=color,
        linewidth=1.8,
    )

if not ax.lines:
    raise FileNotFoundError("No completed participation-rate result curves found.")

ax.set_xlim([0, NUM_ROUNDS])
ax.set_ylim([0.05, 0.60])
ax.set_xticks(np.arange(0, NUM_ROUNDS + 1, 100))
ax.set_yticks(np.arange(0.1, 0.61, 0.1))
ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
ax.legend(handlelength=2.3)

fig.tight_layout()
plt.show()
