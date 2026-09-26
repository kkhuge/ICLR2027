"""Figure 5 right: epsilon 10/20/50 trajectories, seed 0, five-round smoothing."""
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_loss" / "fedavg5"
TOTAL_ROUNDS = 500
SMOOTH_WINDOW = 5
FONT_SIZE = 16
EXPERIMENTS = (
    (10, 0, r"$\epsilon=10$", "tab:red"),
    (20, 0, r"$\epsilon=20$", "tab:orange"),
    (50, 0, r"$\epsilon=50$", "tab:green"),
)

def result_stem(epsilon, seed):
    return (
        f"fmnist_all_data_1_random_iid_lenet_eps{epsilon}_"
        "C0p2_clientdp_sens1_T500_lr0p1_tau5_bs64_constant_"
        f"seed{seed}_epslin"
    )

def load_gap_curve(epsilon, seed):
    stem = result_stem(epsilon, seed)
    train_loss = np.load(RESULT_DIR / f"loss_train_{stem}.npy", allow_pickle=False).astype(float)
    test_loss = np.load(RESULT_DIR / f"loss_test_{stem}.npy", allow_pickle=False).astype(float)
    expected_shape = (TOTAL_ROUNDS + 1,)
    if train_loss.shape != expected_shape or test_loss.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape} for epsilon={epsilon}, seed={seed}")
    # Preserve the original figure's checkpoint selection and x indexing.
    signed_gap = (test_loss - train_loss)[:TOTAL_ROUNDS]
    kernel = np.ones(SMOOTH_WINDOW, dtype=float) / SMOOTH_WINDOW
    smoothed_gap = np.convolve(signed_gap, kernel, mode="valid")
    return np.arange(smoothed_gap.size), smoothed_gap

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(5, 4))
for epsilon, seed, label, color in EXPERIMENTS:
    rounds, gap = load_gap_curve(epsilon, seed)
    ax.plot(rounds, gap, color=color, linestyle="-", linewidth=2.0, label=label)
ax.set_xlabel("Communication Round")
ax.set_ylabel("Generalization Gap")
ax.set_xlim(0, TOTAL_ROUNDS)
ax.set_xticks(np.arange(0, TOTAL_ROUNDS + 1, 100))
ax.set_ylim(0, 0.06)
ax.set_yticks(np.arange(0, 0.061, 0.02))
ax.grid(True, alpha=0.28)
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()
