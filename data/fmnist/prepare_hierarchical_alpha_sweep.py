"""Prepare matched Fashion-MNIST two-level Dirichlet partitions.

All alpha settings use all 60,000 training examples on 500 participating
clients and all 10,000 test examples on 500 participating plus 125 unseen
clients.  Client sizes and global class counts are fixed across alpha values.
The same per-client Dirichlet preference is used for its train and test split.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torchvision


NUM_CLASSES = 10
DEFAULT_ALPHAS = ("0.1", "0.2", "0.3", "0.5", "0.8", "1")


def alpha_tag(alpha):
    return "{:g}".format(float(alpha))


def short_key(alpha, seed):
    key = "fh" + alpha_tag(alpha).replace(".", "p")
    # Keep the existing seed-0 filenames and isolate additional partitions.
    return key if int(seed) == 0 else key + "s{}".format(int(seed))


def as_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def balanced_counts(labels, preferences):
    """Round a preference matrix while preserving rows and class totals."""
    num_clients = len(preferences)
    if len(labels) % num_clients:
        raise ValueError(
            "{} samples are not divisible by {} clients".format(
                len(labels), num_clients
            )
        )
    row_targets = np.full(num_clients, len(labels) // num_clients, dtype=np.int64)
    column_targets = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    expected = np.asarray(preferences, dtype=np.float64).copy()

    for _ in range(1000):
        expected *= row_targets[:, None] / expected.sum(axis=1, keepdims=True)
        expected *= column_targets[None, :] / expected.sum(axis=0, keepdims=True)

    counts = np.floor(expected).astype(np.int64)
    row_deficits = row_targets - counts.sum(axis=1)
    column_deficits = column_targets - counts.sum(axis=0)
    fractional = expected - counts

    for flat_index in np.argsort(fractional, axis=None)[::-1]:
        client_id, class_id = np.unravel_index(flat_index, fractional.shape)
        if row_deficits[client_id] > 0 and column_deficits[class_id] > 0:
            counts[client_id, class_id] += 1
            row_deficits[client_id] -= 1
            column_deficits[class_id] -= 1

    for client_id in np.flatnonzero(row_deficits):
        while row_deficits[client_id] > 0:
            class_id = int(np.argmax(column_deficits))
            if column_deficits[class_id] <= 0:
                raise RuntimeError("Could not round the Dirichlet count matrix")
            counts[client_id, class_id] += 1
            row_deficits[client_id] -= 1
            column_deficits[class_id] -= 1

    if not np.array_equal(counts.sum(axis=1), row_targets):
        raise RuntimeError("Incorrect per-client sample counts")
    if not np.array_equal(counts.sum(axis=0), column_targets):
        raise RuntimeError("Incorrect global class counts")
    return counts


def materialize(images, labels, counts, rng):
    pieces = [[] for _ in range(len(counts))]
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        offsets = np.concatenate(([0], np.cumsum(counts[:, class_id])))
        for client_id in range(len(counts)):
            pieces[client_id].append(indices[offsets[client_id]:offsets[client_id + 1]])

    records = []
    for client_pieces in pieces:
        indices = np.concatenate(client_pieces)
        rng.shuffle(indices)
        records.append({
            "x": images[indices].copy(),
            "y": labels[indices].astype(np.int64, copy=True),
        })
    return records


def payload(user_ids, records, metadata):
    user_data = {user_id: records[index] for index, user_id in enumerate(user_ids)}
    return {
        "users": list(user_ids),
        "user_data": user_data,
        "num_samples": [len(user_data[user_id]["y"]) for user_id in user_ids],
        "hiergen_metadata": metadata,
    }


def write_pickle(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as outfile:
        pickle.dump(value, outfile, pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def prepare_one(root, images, labels, alpha, seed, num_participating, num_unseen):
    total_clients = num_participating + num_unseen
    tag = alpha_tag(alpha)
    key = short_key(alpha, seed)
    preference_rng = np.random.RandomState(seed)
    preferences = preference_rng.dirichlet(
        np.full(NUM_CLASSES, float(alpha)), size=total_clients
    )

    train_counts = balanced_counts(labels["train"], preferences[:num_participating])
    test_counts = balanced_counts(labels["test"], preferences)
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
        "distribution": "Dirichlet label heterogeneity",
        "alpha": float(alpha),
        "partition_seed": int(seed),
        "num_participating_clients": int(num_participating),
        "num_unseen_clients": int(num_unseen),
        "train_samples_per_participating_client": int(len(labels["train"]) // num_participating),
        "test_samples_per_client": int(len(labels["test"]) // total_clients),
        "test_scope": "participating and unseen test only",
        "shared_train_test_preference": True,
    }

    train_path = os.path.join(root, "train", key + ".pkl")
    test_path = os.path.join(root, "test", key + ".pkl")
    unseen_path = os.path.join(root, "hiergen", "unseen_" + key + ".pkl")
    write_pickle(train_path, payload(participating, train_records, metadata))
    write_pickle(test_path, payload(participating, participating_test, metadata))
    write_pickle(unseen_path, payload(unseen, unseen_test, metadata))

    summary = {
        "alpha": float(alpha),
        "key": key,
        "train_path": train_path,
        "test_path": test_path,
        "unseen_path": unseen_path,
        "participating_train_samples": int(len(labels["train"])),
        "participating_test_samples": int(sum(len(x["y"]) for x in participating_test)),
        "unseen_test_samples": int(sum(len(x["y"]) for x in unseen_test)),
        "train_active_classes_mean": float(np.mean((train_counts > 0).sum(axis=1))),
        "test_active_classes_mean": float(np.mean((test_counts > 0).sum(axis=1))),
    }
    print("prepared alpha={}: {}".format(tag, summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--num_participating", type=int, default=500)
    parser.add_argument("--num_unseen", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_participating <= 0 or args.num_unseen <= 0:
        raise ValueError("Client counts must be positive")
    if any(float(alpha) <= 0 for alpha in args.alphas):
        raise ValueError("Dirichlet alpha must be positive")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
    images = {"train": as_numpy(trainset.data), "test": as_numpy(testset.data)}
    labels = {
        "train": as_numpy(trainset.targets).astype(np.int64, copy=False),
        "test": as_numpy(testset.targets).astype(np.int64, copy=False),
    }

    summaries = [
        prepare_one(
            root, images, labels, alpha, args.seed,
            args.num_participating, args.num_unseen,
        )
        for alpha in args.alphas
    ]
    audit_path = os.path.join(root, "hiergen", "partition_audit_fmnist_alpha_seed{}.json".format(args.seed))
    with open(audit_path, "w", encoding="utf-8") as outfile:
        json.dump(summaries, outfile, ensure_ascii=False, indent=2)
    print("audit: {}".format(audit_path), flush=True)


if __name__ == "__main__":
    main()
