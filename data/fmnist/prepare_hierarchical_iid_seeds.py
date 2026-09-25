"""Prepare IID Fashion-MNIST splits for hierarchical generalization.

The official training set is divided among 500 participating clients.  The
official test set is divided among 500 participating-client test sets and 125
unseen clients.  Every client receives an approximately stratified IID class
mixture, while all samples are used exactly once within their source split.
"""

import argparse
import json
import os

import numpy as np
import torchvision

from prepare_hierarchical_alpha_sweep import (
    NUM_CLASSES,
    as_numpy,
    balanced_counts,
    materialize,
    payload,
    write_pickle,
)


def short_key(seed):
    return "fhiid" if int(seed) == 0 else "fhiids{}".format(int(seed))


def iid_preferences(num_clients, seed):
    # Tiny seeded jitter only randomizes integer-rounding ties.  It does not
    # introduce a meaningful class preference.
    rng = np.random.RandomState(seed + 30000)
    return np.ones((num_clients, NUM_CLASSES), dtype=np.float64) + (
        1.0e-9 * rng.random_sample((num_clients, NUM_CLASSES))
    )


def prepare_one(root, images, labels, seed, num_participating, num_unseen):
    total_clients = num_participating + num_unseen
    key = short_key(seed)

    train_counts = balanced_counts(
        labels["train"], iid_preferences(num_participating, seed)
    )
    test_counts = balanced_counts(
        labels["test"], iid_preferences(total_clients, seed + 1000)
    )
    train_records = materialize(
        images["train"], labels["train"], train_counts,
        np.random.RandomState(seed + 10000),
    )
    test_records = materialize(
        images["test"], labels["test"], test_counts,
        np.random.RandomState(seed + 20000),
    )

    participating = list(range(num_participating))
    unseen = list(range(num_participating, total_clients))
    participating_test = test_records[:num_participating]
    unseen_test = test_records[num_participating:]
    metadata = {
        "dataset": "Fashion-MNIST",
        "distribution": "stratified IID",
        "partition_seed": int(seed),
        "num_participating_clients": int(num_participating),
        "num_unseen_clients": int(num_unseen),
        "train_samples_per_participating_client": int(
            len(labels["train"]) // num_participating
        ),
        "test_samples_per_client": int(len(labels["test"]) // total_clients),
        "test_scope": "participating and unseen test only",
    }

    train_path = os.path.join(root, "train", key + ".pkl")
    test_path = os.path.join(root, "test", key + ".pkl")
    unseen_path = os.path.join(root, "hiergen", "unseen_" + key + ".pkl")
    write_pickle(train_path, payload(participating, train_records, metadata))
    write_pickle(test_path, payload(participating, participating_test, metadata))
    write_pickle(unseen_path, payload(unseen, unseen_test, metadata))

    summary = {
        "seed": int(seed),
        "key": key,
        "train_path": train_path,
        "test_path": test_path,
        "unseen_path": unseen_path,
        "participating_train_samples": int(len(labels["train"])),
        "participating_test_samples": int(
            sum(len(record["y"]) for record in participating_test)
        ),
        "unseen_test_samples": int(
            sum(len(record["y"]) for record in unseen_test)
        ),
        "train_active_classes_min": int((train_counts > 0).sum(axis=1).min()),
        "test_active_classes_min": int((test_counts > 0).sum(axis=1).min()),
    }
    audit_path = os.path.join(
        root, "hiergen", "partition_audit_fmnist_iid_seed{}.json".format(seed)
    )
    with open(audit_path, "w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)
    print("prepared IID seed={}: {}".format(seed, summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_participating", type=int, default=500)
    parser.add_argument("--num_unseen", type=int, default=125)
    args = parser.parse_args()

    if args.num_participating <= 0 or args.num_unseen <= 0:
        raise ValueError("Client counts must be positive")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=False)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=False)
    images = {"train": as_numpy(trainset.data), "test": as_numpy(testset.data)}
    labels = {
        "train": as_numpy(trainset.targets).astype(np.int64, copy=False),
        "test": as_numpy(testset.targets).astype(np.int64, copy=False),
    }

    for seed in args.seeds:
        prepare_one(
            root, images, labels, seed,
            args.num_participating, args.num_unseen,
        )


if __name__ == "__main__":
    main()
