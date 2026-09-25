"""Plot Fashion-MNIST training loss for the eta_0=0.1 LR sweep."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

apply_plot_style()

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_loss" / "fedavg5"
TOTAL_ROUNDS = 500
PLOT_ROUNDS = 300
SMOOTH_WINDOW = 5

EXPERIMENTS = (
    ("a0", r"$a=0$", "tab:red"),
    ("a0p5", r"$a=0.5$", "tab:orange"),
    ("a0p75", r"$a=0.75$", "tab:green"),
    ("a1", r"$a=1$", "tab:cyan"),
)


def result_path(suffix: str) -> Path:
    stem = (
        "fmnist_all_data_1_random_iid_lenet_"
        "sigma0p9_C0p2_clientdp_sens1_T500_"
        f"lr0p1_tau5_bs64_power_seed0_{suffix}"
    )
    return RESULT_DIR / f"loss_train_{stem}.npy"


def load_curve(suffix: str) -> np.ndarray:
    path = result_path(suffix)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    values = np.load(path, allow_pickle=False).astype(float)
    if values.ndim != 1 or len(values) != TOTAL_ROUNDS + 1:
        raise ValueError(
            f"Expected {TOTAL_ROUNDS + 1} values in {path}, got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {path}.")
    return values


def smooth(values: np.ndarray) -> np.ndarray:
    kernel = np.ones(SMOOTH_WINDOW, dtype=float) / SMOOTH_WINDOW
    return np.convolve(values, kernel, mode="valid")


rounds = np.arange(PLOT_ROUNDS - SMOOTH_WINDOW + 1)
fig, ax = plt.subplots(figsize=(5, 4))

for suffix, label, color in EXPERIMENTS:
    training_loss = smooth(load_curve(suffix)[:PLOT_ROUNDS])
    ax.plot(
        rounds,
        training_loss,
        label=label,
        color=color,
        linestyle="-",
        linewidth=2.0,
    )

ax.set_xlabel("Communication Round")
ax.set_ylabel("Training Loss")
ax.set_xlim(0, PLOT_ROUNDS)
ax.set_yticks(np.arange(0, 2.51, 0.5))
ax.set_ylim(0, 2.5)
ax.set_xticks(np.arange(0, PLOT_ROUNDS + 1, 50))
ax.grid(True, alpha=0.28)
ax.legend(handlelength=2.3)
fig.tight_layout()

plt.show()
