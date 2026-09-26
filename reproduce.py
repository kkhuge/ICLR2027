"""Generate the data, runs, and plots used by the paper's 18 figure panels.

Run from any directory. Commands are executed from this file's directory so
the legacy trainer's relative data and result paths remain valid.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GROUPS = (
    "femnist", "heterogeneity", "local_size", "learning_rate", "privacy",
    "cifar_sigma", "cifar_clip", "cifar_participation", "cifar_tau",
)
PLOTS = {
    "femnist": ("loss_femnist_niid_client_level.py", "acc_femnist_niid_client_level.py"),
    "heterogeneity": ("gap1_fmnist_niid.py", "gap2_fmnist_niid.py"),
    "local_size": ("gap1_fmnist_local_datasize.py", "gap2_fmnist_local_datasize.py"),
    "learning_rate": ("fmnist_training_loss_eta.py", "fmnist_gap_eta.py"),
    "privacy": ("gap_fmnist_epsilon.py", "gap_fmnist_epsilon_vary.py"),
    "cifar_sigma": ("cifar10_acc_sigma.py", "cifar10_gap_sigma.py"),
    "cifar_clip": ("cifar10_acc_clip.py", "cifar10_gap_clip.py"),
    "cifar_participation": ("cifar10_acc_client_number.py", "cifar10_gap_client_number.py"),
    "cifar_tau": ("cifar10_acc_tau.py", "cifar10_gap_tau.py"),
}


@dataclass(frozen=True)
class Run:
    dataset: str
    rounds: int
    suffix: str
    seed: int = 0
    sigma: float | None = 0.9
    epsilon: int | None = None
    clip: float = 0.2
    clients: int = 50
    epochs: int = 5
    schedule: str = "constant"
    decay: float | None = None
    unseen: str | None = None
    setting: str | None = None
    metric_weighting: str = "equal"
    update_weighting: str = "equal"

    @property
    def hierarchy(self) -> bool:
        return self.unseen is not None

    @property
    def stem(self) -> str:
        def tag(number):
            return f"{number:g}".replace(".", "p")

        privacy = f"eps{self.epsilon}" if self.epsilon is not None else f"sigma{tag(self.sigma)}"
        return (
            f"{self.dataset}_lenet_{privacy}_C{tag(self.clip)}_clientdp_sens1_"
            f"T{self.rounds}_lr0p1_tau{self.epochs}_bs64_{self.schedule}_"
            f"seed{self.seed}_{self.suffix}"
        )

    @property
    def outputs(self) -> list[Path]:
        if self.hierarchy:
            folder = ROOT / "result_hiergen" / "fedavg5"
            metrics = (
                "loss_participating_train", "loss_participating_val", "loss_unseen_clients",
                "acc_participating_train", "acc_participating_val", "acc_unseen_clients",
            )
            return [folder / f"{metric}_{self.stem}.npy" for metric in metrics]
        loss = ROOT / "result_loss" / "fedavg5"
        accuracy = ROOT / "result_acc" / "fedavg5"
        return [
            loss / f"loss_train_{self.stem}.npy",
            loss / f"loss_test_{self.stem}.npy",
            accuracy / f"acc_train_{self.stem}.npy",
            accuracy / f"acc_test_{self.stem}.npy",
        ]

    def is_complete(self) -> bool:
        for path in self.outputs:
            if not path.exists():
                return False
            try:
                values = np.load(path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError):
                return False
            if values.shape != (self.rounds + 1,):
                return False
        return True

    def command(self, python: str) -> list[str]:
        cmd = [
            python, "main.py", "--algo", "fedavg_hiergen" if self.hierarchy else "fedavg5",
            "--dataset", self.dataset, "--model", "lenet", "--num_round", str(self.rounds),
            "--eval_every", "1", "--clients_per_round", str(self.clients),
            "--batch_size", "64", "--num_epoch", str(self.epochs), "--lr", "0.1",
            "--wd", "0", "--dp_clip_norm", str(self.clip), "--dp_delta", "0.002",
            "--lr_schedule", self.schedule, "--seed", str(self.seed),
            "--result_suffix", self.suffix, "--progress_every", "20", "--noprint",
        ]
        if self.epsilon is not None:
            cmd += ["--target_epsilon", str(self.epsilon)]
        else:
            cmd += ["--dp_noise_multiplier", str(self.sigma)]
        if self.decay is not None:
            cmd += ["--lr_decay_exponent", str(self.decay)]
        if self.hierarchy:
            cmd += ["--hiergen_unseen_file", self.unseen, "--hiergen_setting", self.setting]
            cmd += [
                "--hiergen_metric_weighting", self.metric_weighting,
                "--hiergen_update_weighting", self.update_weighting,
            ]
        return cmd


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def runs_by_group() -> dict[str, list[Run]]:
    runs = {group: [] for group in GROUPS}
    runs["femnist"].append(Run(
        "femnist_hn0", 2000, "n",
        unseen="data/femnist/data/hiergen/unseen_hn0.pkl", setting="natural_niid",
    ))

    # These are the selected runs actually read by both heterogeneity panels.
    for alpha, seed, rounds in ((0.5, 1, 2000), (0.8, 0, 1200), (1.0, 2, 1200)):
        key = f"fh{tag(alpha)}" + (f"s{seed}" if seed else "")
        suffix = f"a{tag(alpha)}" + (f"s{seed}" if seed else "")
        runs["heterogeneity"].append(Run(
            f"fmnist_{key}", rounds, suffix, seed=seed,
            unseen=f"data/fmnist/data/hiergen/unseen_{key}.pkl", setting="natural_niid",
        ))
    runs["heterogeneity"].append(Run(
        "fmnist_fhiids2", 1200, "iids2", seed=2,
        unseen="data/fmnist/data/hiergen/unseen_fhiids2.pkl", setting="iid",
    ))

    for size in (80, 100, 120):
        key = f"fhls{size}s2"
        runs["local_size"].append(Run(
            f"fmnist_{key}", 1200, f"n{size}s2", seed=2,
            unseen=f"data/fmnist/data/hiergen/unseen_{key}.pkl",
            setting="iid", metric_weighting="samples", update_weighting="equal",
        ))

    for label, exponent in (("a0", 0), ("a0p5", 0.5), ("a0p75", 0.75), ("a1", 1)):
        runs["learning_rate"].append(Run(
            "fmnist_all_data_1_random_iid", 500, label,
            schedule="power", decay=exponent,
        ))

    for rounds in (400, 500, 600, 700, 800):
        for seed in range(5):
            runs["privacy"].append(Run(
                "fmnist_all_data_1_random_iid", rounds, "pc16",
                seed=seed, sigma=None, epsilon=16,
            ))
    for epsilon in (6, 8, 10):
        for rounds in (500, 600, 700, 800):
            for seed in range(5):
                runs["privacy"].append(Run(
                    "fmnist_all_data_1_random_iid", rounds, "epslin",
                    seed=seed, sigma=None, epsilon=epsilon,
                ))

    # Figure 5 right uses T=500, seed=0; epsilon=10 is included above.
    for epsilon in (20, 50):
        runs["privacy"].append(Run(
            "fmnist_all_data_1_random_iid", 500, "epslin",
            seed=0, sigma=None, epsilon=epsilon,
        ))

    default = Run("cifar10_all_data_1_random_iid", 500, "lenett500s0p9c0p2")
    for group in ("cifar_sigma", "cifar_tau"):
        runs[group].append(default)
    for sigma in (0.6, 1.2, 1.5):
        runs["cifar_sigma"].append(Run(
            default.dataset, 500, f"lenett500c0p2s{tag(sigma)}", sigma=sigma,
        ))
    for clip in (0.3, 0.4, 0.5):
        runs["cifar_clip"].append(Run(
            default.dataset, 500, f"lenett500s0p9c{tag(clip)}", clip=clip,
        ))
    for clients in (100, 200, 500):
        runs["cifar_participation"].append(Run(
            default.dataset, 500, f"lenett500s0p9c0p2m{clients}", clients=clients,
        ))
    for epochs, seed in ((1, 1), (2, 2), (10, 0)):
        suffix = f"lenett500s0p9c0p2tau{epochs}" + (f"seed{seed}" if seed else "")
        runs["cifar_tau"].append(Run(
            default.dataset, 500, suffix, seed=seed, epochs=epochs,
        ))
    return runs


def execute(cmd: list[str], dry_run: bool) -> None:
    print("$", subprocess.list2cmdline(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def prepare(group: str, python: str, dry_run: bool) -> None:
    helper = ROOT / "data/mnist/data/train/all_data_1_random_iid.pkl"
    if not helper.exists():
        execute([python, "data/mnist/generate_random_iid.py"], dry_run)

    if group == "femnist":
        execute([python, "download_femnist.py"], dry_run)
        execute([
            python, "data/femnist/prepare_client_level_hierarchical.py",
            "--seed", "0", "--local_split", "tff_original",
            "--unseen_evaluation", "test_only",
        ], dry_run)
    elif group == "heterogeneity":
        for alpha, seed in ((0.5, 1), (0.8, 0), (1.0, 2)):
            execute([
                python, "data/fmnist/prepare_hierarchical_alpha_sweep.py",
                "--alphas", f"{alpha:g}", "--seed", str(seed),
            ], dry_run)
        execute([python, "data/fmnist/prepare_hierarchical_iid_seeds.py", "--seeds", "2"], dry_run)
    elif group == "local_size":
        execute([
            python, "data/fmnist/prepare_hierarchical_local_size_sweep.py",
            "--sizes", "80", "100", "120", "--seed", "2",
        ], dry_run)
    elif group in ("learning_rate", "privacy"):
        execute([python, "data/fmnist/generate_iid.py"], dry_run)
    else:
        execute([python, "data/cifar10/generate_cifar_iid.py"], dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all",) + GROUPS, default="all")
    parser.add_argument("--stage", choices=("data", "train", "plot", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Rerun existing training outputs")
    args = parser.parse_args()
    groups = GROUPS if args.group == "all" else (args.group,)
    runs = runs_by_group()
    completed = set()
    for group in groups:
        print(f"\n== {group} ==", flush=True)
        if args.stage in ("data", "all"):
            prepare(group, sys.executable, args.dry_run)
        if args.stage in ("train", "all"):
            for run in runs[group]:
                if run.stem in completed:
                    continue
                completed.add(run.stem)
                if run.is_complete() and not args.force:
                    print(f"Already complete: {run.stem}", flush=True)
                    continue
                execute(run.command(sys.executable), args.dry_run)
        if args.stage in ("plot", "all"):
            for plot in PLOTS[group]:
                execute([sys.executable, "plot_runner.py", plot], args.dry_run)


if __name__ == "__main__":
    main()
