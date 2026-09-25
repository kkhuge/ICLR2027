"""Plot natural-NIID FEMNIST training, seen-client, and unseen-client accuracies."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

apply_plot_style()


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_hiergen" / "fedavg5"
STEM = (
    "femnist_hn0_lenet_sigma0p9_C0p2_clientdp_sens1_T2000_"
    "lr0p1_tau5_bs64_constant_seed0_n"
)
SMOOTH = 1
MAX_ROUND = 1200
CURVES = (
    ("acc_participating_train", "Training accuracy", "tab:red"),
    ("acc_participating_val", "Seen-client accuracy", "tab:orange"),
    ("acc_unseen_clients", "Unseen-client accuracy", "tab:green"),
)


def load_curve(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = RESULT_DIR / f"{name}_{STEM}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")

    values = np.load(path, allow_pickle=False).astype(float)
    if values.ndim != 1 or len(values) < MAX_ROUND + 1:
        raise ValueError(
            f"Expected a 1-D curve containing rounds 0--{MAX_ROUND} in "
            f"{path}, got shape {values.shape}"
        )

    values = values[: MAX_ROUND + 1]
    kernel = np.ones(SMOOTH, dtype=float) / SMOOTH
    smoothed = np.convolve(values, kernel, mode="valid")
    rounds = np.arange(smoothed.size)
    return rounds, smoothed


plt.figure(figsize=(5, 4))
for name, label, color in CURVES:
    rounds, accuracy = load_curve(name)
    plt.plot(
        rounds,
        accuracy,
        label=label,
        color=color,
        linestyle="-",
        linewidth=2.0,
    )

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.xlim(0, MAX_ROUND)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, MAX_ROUND + 1, 400))
plt.yticks(np.arange(0.0, 1.01, 0.2))
plt.grid(True, alpha=0.28)
plt.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.show()
