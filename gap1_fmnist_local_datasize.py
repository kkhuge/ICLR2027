"""Plot Fashion-MNIST IID local-size within-client gaps through round 1200."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

apply_plot_style()


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_hiergen" / "fedavg5"
TOTAL_ROUNDS = 1200
SMOOTH_WINDOW = 20
SEED = 2

EXPERIMENTS = (
    (80, r"$|\mathcal{D}_{i}|=80$", "tab:red"),
    (100, r"$|\mathcal{D}_{i}|=100$", "tab:orange"),
    (120, r"$|\mathcal{D}_{i}|=120$", "tab:green"),
)


def experiment_spec(size):
    return (
        f"fmnist_fhls{size}s{SEED}_lenet_sigma0p9_C0p2_clientdp_sens1_"
        f"T1200_lr0p1_tau5_bs64_constant_seed{SEED}_n{size}s{SEED}"
    )


def load_curve(metric, size):
    path = RESULT_DIR / f"{metric}_{experiment_spec(size)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    values = np.load(path, allow_pickle=False).astype(float)
    if values.ndim != 1 or values.size < TOTAL_ROUNDS + 1:
        raise ValueError(f"Expected at least 1201 values in {path}, got {values.shape}.")
    return values[: TOTAL_ROUNDS + 1]


def smooth(values):
    values = np.asarray(values, dtype=float)[:TOTAL_ROUNDS]
    kernel = np.ones(SMOOTH_WINDOW, dtype=float) / SMOOTH_WINDOW
    return np.convolve(values, kernel, mode="valid")


rounds = np.arange(TOTAL_ROUNDS - SMOOTH_WINDOW + 1)
fig, ax = plt.subplots(figsize=(5, 4))

for size, label, color in EXPERIMENTS:
    train_loss = load_curve("loss_participating_train", size)
    seen_loss = load_curve("loss_participating_val", size)
    ax.plot(
        rounds,
        smooth(seen_loss - train_loss),
        label=label,
        color=color,
        linestyle="-",
        linewidth=2.0,
    )

ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
ax.set_xlabel("Communication Round")
ax.set_ylabel("Within-Client Gen")
ax.set_xlim(0, TOTAL_ROUNDS)
ax.set_xticks(np.arange(0, TOTAL_ROUNDS + 1, 300))
ax.set_ylim(0, 0.15)
ax.set_yticks(np.arange(0, 0.151, 0.05))
ax.grid(True, alpha=0.28)
ax.legend(handlelength=2)
fig.tight_layout()
plt.show()
