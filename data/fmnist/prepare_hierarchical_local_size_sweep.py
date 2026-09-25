"""Prepare nested IID Fashion-MNIST partitions for a local-size sweep."""

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


DEFAULT_SIZES = (80, 60, 50, 20)


def short_key(size, seed):
    return "fhls{}s{}".format(int(size), int(seed))


def build_nested_indices(labels, sizes, num_clients, seed):
    max_size = max(sizes)
    per_class_max = max_size // NUM_CLASSES
    class_blocks = []
    rng = np.random.RandomState(seed + 10000)

    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        needed = num_clients * per_class_max
        if needed > len(indices):
            raise ValueError(
                "Need {} class-{} examples, but Fashion-MNIST has only {}".format(
                    needed, class_id, len(indices)
                )
            )
        class_blocks.append(indices[:needed].reshape(num_clients, per_class_max))

    indices_by_size = {}
    for size in sizes:
        per_class = size // NUM_CLASSES
        clients = []
        for client_id in range(num_clients):
            selected = np.concatenate(
                [block[client_id, :per_class] for block in class_blocks]
            )
            np.random.RandomState(seed + size * 1000 + client_id).shuffle(selected)
            clients.append(selected)
        indices_by_size[size] = clients
    return indices_by_size


def validate_indices(labels, indices_by_size, sizes, num_clients):
    descending = sorted(sizes, reverse=True)
    largest = descending[0]
    largest_sets = [set(x.tolist()) for x in indices_by_size[largest]]
    if len(set().union(*largest_sets)) != largest * num_clients:
        raise RuntimeError("Training examples overlap across clients")

    for size in sizes:
        expected = np.full(NUM_CLASSES, size // NUM_CLASSES)
        for client_id, indices in enumerate(indices_by_size[size]):
            if len(indices) != size:
                raise RuntimeError("Incorrect sample count for client {}".format(client_id))
            if not np.array_equal(
                np.bincount(labels[indices], minlength=NUM_CLASSES), expected
            ):
                raise RuntimeError("Client {} is not class-balanced".format(client_id))

    for larger, smaller in zip(descending, descending[1:]):
        for client_id in range(num_clients):
            if not set(indices_by_size[smaller][client_id].tolist()).issubset(
                set(indices_by_size[larger][client_id].tolist())
            ):
                raise RuntimeError(
                    "n={} is not nested in n={} for client {}".format(
                        smaller, larger, client_id
                    )
                )


def records_from_indices(images, labels, client_indices):
    return [
        {
            "x": images[indices].copy(),
            "y": labels[indices].astype(np.int64, copy=True),
        }
        for indices in client_indices
    ]


def iid_preferences(num_clients, seed):
    rng = np.random.RandomState(seed + 30000)
    return np.ones((num_clients, NUM_CLASSES), dtype=np.float64) + (
        1.0e-9 * rng.random_sample((num_clients, NUM_CLASSES))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--num_participating", type=int, default=500)
    parser.add_argument("--num_unseen", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    sizes = tuple(dict.fromkeys(args.sizes))
    if args.num_participating <= 0 or args.num_unseen <= 0:
        raise ValueError("Client counts must be positive")
    if not sizes or any(size <= 0 or size % NUM_CLASSES for size in sizes):
        raise ValueError("Every local size must be positive and divisible by 10")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
    train_images = as_numpy(trainset.data)
    train_labels = as_numpy(trainset.targets).astype(np.int64, copy=False)
    test_images = as_numpy(testset.data)
    test_labels = as_numpy(testset.targets).astype(np.int64, copy=False)

    indices_by_size = build_nested_indices(
        train_labels, sizes, args.num_participating, args.seed
    )
    validate_indices(train_labels, indices_by_size, sizes, args.num_participating)

    total_clients = args.num_participating + args.num_unseen
    test_counts = balanced_counts(
        test_labels, iid_preferences(total_clients, args.seed + 1000)
    )
    test_records = materialize(
        test_images,
        test_labels,
        test_counts,
        np.random.RandomState(args.seed + 20000),
    )
    participating = list(range(args.num_participating))
    unseen = list(range(args.num_participating, total_clients))
    participating_test = test_records[: args.num_participating]
    unseen_test = test_records[args.num_participating :]

    summaries = []
    for size in sizes:
        key = short_key(size, args.seed)
        metadata = {
            "dataset": "Fashion-MNIST",
            "distribution": "stratified IID class-balanced",
            "partition_seed": int(args.seed),
            "num_participating_clients": int(args.num_participating),
            "num_unseen_clients": int(args.num_unseen),
            "train_samples_per_participating_client": int(size),
            "train_samples_per_class_per_client": int(size // NUM_CLASSES),
            "test_samples_per_client": int(len(test_labels) // total_clients),
            "test_scope": "participating and unseen test only",
            "nested_local_training_subsets": True,
        }
        train_records = records_from_indices(
            train_images, train_labels, indices_by_size[size]
        )
        train_path = os.path.join(root, "train", key + ".pkl")
        test_path = os.path.join(root, "test", key + ".pkl")
        unseen_path = os.path.join(root, "hiergen", "unseen_" + key + ".pkl")
        write_pickle(train_path, payload(participating, train_records, metadata))
        write_pickle(test_path, payload(participating, participating_test, metadata))
        write_pickle(unseen_path, payload(unseen, unseen_test, metadata))

        summary = {
            "local_train_samples_per_client": int(size),
            "key": key,
            "train_path": train_path,
            "test_path": test_path,
            "unseen_path": unseen_path,
            "participating_train_samples": int(size * args.num_participating),
            "participating_test_samples": int(
                sum(len(record["y"]) for record in participating_test)
            ),
            "unseen_test_samples": int(
                sum(len(record["y"]) for record in unseen_test)
            ),
            "class_histogram_per_train_client": [int(size // NUM_CLASSES)] * NUM_CLASSES,
        }
        summaries.append(summary)
        print("prepared n={}: {}".format(size, summary), flush=True)

    audit_path = os.path.join(
        root,
        "hiergen",
        "partition_audit_fmnist_hiergen_local_size_seed{}.json".format(args.seed),
    )
    with open(audit_path, "w", encoding="utf-8") as outfile:
        json.dump(summaries, outfile, ensure_ascii=False, indent=2)
    print("audit: {}".format(audit_path), flush=True)


if __name__ == "__main__":
    main()
