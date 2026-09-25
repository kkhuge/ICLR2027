"""Plot Fashion-MNIST client-level gaps for the eta_0=0.1 LR sweep."""

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


def result_path(metric: str, suffix: str) -> Path:
    stem = (
        "fmnist_all_data_1_random_iid_lenet_"
        "sigma0p9_C0p2_clientdp_sens1_T500_"
        f"lr0p1_tau5_bs64_power_seed0_{suffix}"
    )
    return RESULT_DIR / f"{metric}_{stem}.npy"


def load_curve(metric: str, suffix: str) -> np.ndarray:
    path = result_path(metric, suffix)
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
    train_loss = load_curve("loss_train", suffix)
    test_loss = load_curve("loss_test", suffix)

    # Signed empirical gap; negative values are intentionally retained.
    gap = smooth((test_loss - train_loss)[:PLOT_ROUNDS])
    ax.plot(
        rounds,
        gap,
        label=label,
        color=color,
        linestyle="-",
        linewidth=2.0,
    )


ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Loss - Training Loss")
ax.set_xlim(0, PLOT_ROUNDS)
ax.set_xticks(np.arange(0, PLOT_ROUNDS + 1, 50))
ax.set_ylim(0, 0.05)
ax.set_yticks(np.arange(0, 0.051, 0.01))
ax.grid(True, alpha=0.28)
ax.legend(handlelength=2.3)
fig.tight_layout()

plt.show()
