"""Plot the five-seed final-20-round mean gap against training horizon."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

apply_plot_style()

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_loss" / "fedavg5"
TRAINING_HORIZONS = (400, 500, 600, 700, 800)
SEEDS = (0, 1, 2, 3, 4)
TAIL_ROUNDS = 20


def result_stem(rounds, seed):
    return (
        "fmnist_all_data_1_random_iid_lenet_eps16_C0p2_clientdp_sens1_"
        f"T{rounds}_lr0p1_tau5_bs64_constant_seed{seed}_pc16"
    )


def load_seed_final_mean_gap(rounds, seed):
    stem = result_stem(rounds, seed)
    train_path = RESULT_DIR / f"loss_train_{stem}.npy"
    test_path = RESULT_DIR / f"loss_test_{stem}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing result for T={rounds}, seed={seed}")

    train_loss = np.load(train_path, allow_pickle=False).astype(float)
    test_loss = np.load(test_path, allow_pickle=False).astype(float)
    expected_shape = (rounds + 1,)
    if train_loss.shape != expected_shape or test_loss.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} for T={rounds}, seed={seed}, got "
            f"{train_loss.shape} and {test_loss.shape}"
        )

    gap = test_loss - train_loss
    return float(np.mean(gap[-TAIL_ROUNDS:]))


seed_gaps = {
    rounds: np.asarray(
        [load_seed_final_mean_gap(rounds, seed) for seed in SEEDS],
        dtype=float,
    )
    for rounds in TRAINING_HORIZONS
}
mean_gaps = np.asarray(
    [np.mean(seed_gaps[rounds]) for rounds in TRAINING_HORIZONS], dtype=float
)
plot_positions = np.arange(len(TRAINING_HORIZONS))

for rounds, gap in zip(TRAINING_HORIZONS, mean_gaps):
    values = ", ".join(f"{value:.6f}" for value in seed_gaps[rounds])
    print(
        f"T={rounds}: seed final-{TAIL_ROUNDS}-round gaps = [{values}], "
        f"five-seed mean = {gap:.6f}"
    )

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(
    plot_positions,
    mean_gaps,
    color="tab:red",
    linestyle="-",
    linewidth=2.0,
    marker="o",
    markersize=5.0,
)

ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Loss - Training Loss")
ax.set_xlim(-0.4, len(TRAINING_HORIZONS) - 0.6)
ax.set_xticks(plot_positions, labels=TRAINING_HORIZONS)
ax.set_ylim(0, 0.08)
ax.set_yticks(np.arange(0, 0.081, 0.02))
ax.grid(True, alpha=0.28)
fig.tight_layout()

plt.show()
