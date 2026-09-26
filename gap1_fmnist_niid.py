"""Plot selected Fashion-MNIST within-client gaps through round 1200.

Selection from completed runs:
alpha=0.5 -> seed 1, alpha=0.8 -> seed 0, alpha=1.0 -> seed 2.
"""

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

ALPHA_EXPERIMENTS = (
    ("0p5", 1, r"$\alpha=0.5$", "tab:red"),
    ("0p8", 0, r"$\alpha=0.8$", "tab:orange"),
    ("1", 2, r"$\alpha=1$", "tab:green"),
)
IID_EXPERIMENTS = (("iid", 2, "IID", "tab:cyan"),)
EXPERIMENTS = ALPHA_EXPERIMENTS + IID_EXPERIMENTS


def experiment_spec(alpha_key, seed):
    if alpha_key == "iid":
        dataset_key = "fhiid" if seed == 0 else f"fhiids{seed}"
        suffix = "iid" if seed == 0 else f"iids{seed}"
        return (
            f"fmnist_{dataset_key}_lenet_sigma0p9_C0p2_clientdp_sens1_"
            f"T1200_lr0p1_tau5_bs64_constant_seed{seed}_{suffix}"
        )

    dataset_key = f"fh{alpha_key}" if seed == 0 else f"fh{alpha_key}s{seed}"
    suffix = f"a{alpha_key}" if seed == 0 else f"a{alpha_key}s{seed}"
    run_rounds = 2000 if seed == 1 else 1200
    stem = (
        f"fmnist_{dataset_key}_lenet_sigma0p9_C0p2_clientdp_sens1_"
        f"T{run_rounds}_lr0p1_tau5_bs64_constant_seed{seed}_{suffix}"
    )
    return stem


def load_curve(metric, alpha_key, seed):
    path = RESULT_DIR / f"{metric}_{experiment_spec(alpha_key, seed)}.npy"
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

for alpha_key, seed, label, color in EXPERIMENTS:
    train_loss = load_curve("loss_participating_train", alpha_key, seed)
    seen_loss = load_curve("loss_participating_val", alpha_key, seed)
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
ax.set_ylim(0, 0.12)
ax.set_yticks(np.arange(0, 0.13, 0.02))
ax.grid(True, alpha=0.28)
ax.legend(
    handlelength=2.3,
)
fig.tight_layout()

plt.show()
