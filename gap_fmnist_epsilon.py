"""Compare privacy budgets using common-seed final-20-round mean gaps."""

from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_loss" / "fedavg5"
TRAINING_HORIZONS = (500, 600, 700, 800)
CANDIDATE_SEEDS = (0, 1, 2, 3, 4)
EPSILON_CONFIGS = (
    (10, "tab:red", "o"),
    (8, "tab:orange", "s"),
    (6, "tab:green", "^"),
)
TAIL_ROUNDS = 20
FONT_SIZE = 16


def result_stem(epsilon, rounds, seed):
    return (
        f"fmnist_all_data_1_random_iid_lenet_eps{epsilon}_"
        "C0p2_clientdp_sens1_"
        f"T{rounds}_lr0p1_tau5_bs64_constant_seed{seed}_epslin"
    )


def load_seed_final_mean_gap(epsilon, rounds, seed):
    stem = result_stem(epsilon, rounds, seed)
    train_path = RESULT_DIR / f"loss_train_{stem}.npy"
    test_path = RESULT_DIR / f"loss_test_{stem}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing result for epsilon={epsilon}, T={rounds}, seed={seed}"
        )

    train_loss = np.load(train_path, allow_pickle=False).astype(float)
    test_loss = np.load(test_path, allow_pickle=False).astype(float)
    expected_shape = (rounds + 1,)
    if train_loss.shape != expected_shape or test_loss.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape} for epsilon={epsilon}, "
            f"T={rounds}, seed={seed}, got "
            f"{train_loss.shape} and {test_loss.shape}"
        )

    gap = test_loss - train_loss
    return float(np.mean(gap[-TAIL_ROUNDS:]))


def seed_has_complete_grid(seed):
    """Require this seed for every epsilon and every training horizon."""
    for epsilon, _, _ in EPSILON_CONFIGS:
        for rounds in TRAINING_HORIZONS:
            stem = result_stem(epsilon, rounds, seed)
            paths = (
                RESULT_DIR / f"loss_train_{stem}.npy",
                RESULT_DIR / f"loss_test_{stem}.npy",
            )
            for path in paths:
                if not path.exists():
                    return False
                try:
                    values = np.load(path, allow_pickle=False, mmap_mode="r")
                except (OSError, ValueError):
                    return False
                if values.shape != (rounds + 1,):
                    return False
    return True


SEEDS = tuple(seed for seed in CANDIDATE_SEEDS if seed_has_complete_grid(seed))
if not SEEDS:
    raise RuntimeError("No seed has a complete epsilon-by-horizon result grid.")
print(f"Averaging common complete seeds: {SEEDS}")


def load_epsilon_mean_gaps(epsilon):
    seed_gaps = {
        rounds: np.asarray(
            [
                load_seed_final_mean_gap(epsilon, rounds, seed)
                for seed in SEEDS
            ],
            dtype=float,
        )
        for rounds in TRAINING_HORIZONS
    }
    mean_gaps = np.asarray(
        [np.mean(seed_gaps[rounds]) for rounds in TRAINING_HORIZONS],
        dtype=float,
    )
    return seed_gaps, mean_gaps

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
    }
)

fig, ax = plt.subplots(figsize=(5, 4))
plot_positions = np.arange(len(TRAINING_HORIZONS))
for epsilon, color, marker in EPSILON_CONFIGS:
    try:
        seed_gaps, mean_gaps = load_epsilon_mean_gaps(epsilon)
    except FileNotFoundError as error:
        print(f"Skipping epsilon={epsilon}: {error}")
        continue

    for rounds, gap in zip(TRAINING_HORIZONS, mean_gaps):
        values = ", ".join(f"{value:.6f}" for value in seed_gaps[rounds])
        print(
            f"epsilon={epsilon}, T={rounds}: "
            f"seed final-{TAIL_ROUNDS}-round gaps = [{values}], "
            f"{len(SEEDS)}-seed mean = {gap:.6f}"
        )

    ax.plot(
        plot_positions,
        mean_gaps,
        color=color,
        linestyle="-",
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        label=rf"$\epsilon={epsilon}$",
    )

ax.set_xlabel("Communication Round")
ax.set_ylabel("Test Loss - Training Loss")
ax.set_xlim(-0.2, plot_positions[-1])
ax.set_xticks(plot_positions, labels=TRAINING_HORIZONS)
ax.set_ylim(0, 0.08)
ax.set_yticks(np.arange(0, 0.081, 0.02))
ax.grid(True, alpha=0.28)
ax.legend(handlelength=2.3)
fig.tight_layout()

plt.show()
