import os
import pickle

import numpy as np
import torchvision


CPATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CPATH, "data")

NUM_USER = 500
NUM_CLASSES = 10
SEED = 6
SAVE = True

TRAIN_FILENAME = "all_data_1_random_iid.pkl"
TEST_FILENAME = "all_data_1_random_iid.pkl"


def _as_numpy(value):
    """Convert torchvision tensors/label containers to NumPy arrays."""
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _iid_partition(images, labels, num_users, rng):
    """Split every class evenly so every client has the same class balance."""
    client_indices = [[] for _ in range(num_users)]

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        # np.array_split keeps every sample even when a class size is not
        # exactly divisible by the number of clients.
        for user_id, split in enumerate(np.array_split(class_indices, num_users)):
            client_indices[user_id].append(split)

    client_images = []
    client_labels = []
    for parts in client_indices:
        indices = np.concatenate(parts)
        rng.shuffle(indices)
        client_images.append(images[indices].copy())
        client_labels.append(labels[indices].astype(np.int64, copy=True))

    return client_images, client_labels


def _build_payload(client_images, client_labels):
    payload = {"users": [], "user_data": {}, "num_samples": []}

    for user_id, (images, labels) in enumerate(zip(client_images, client_labels)):
        payload["users"].append(user_id)
        payload["user_data"][user_id] = {"x": images, "y": labels}
        payload["num_samples"].append(len(labels))

    return payload


def _validate_partition(payload, expected_total, split_name):
    sizes = np.asarray(payload["num_samples"])
    labels = np.concatenate(
        [payload["user_data"][user_id]["y"] for user_id in payload["users"]]
    )

    if len(payload["users"]) != NUM_USER:
        raise RuntimeError(
            "{} split has {} clients, expected {}.".format(
                split_name, len(payload["users"]), NUM_USER
            )
        )
    if int(sizes.sum()) != expected_total:
        raise RuntimeError(
            "{} split contains {} samples, expected {}.".format(
                split_name, int(sizes.sum()), expected_total
            )
        )
    if sizes.min() <= 0:
        raise RuntimeError("{} split contains an empty client.".format(split_name))
    if set(np.unique(labels).tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError("{} split does not contain all classes.".format(split_name))

    print(
        ">>> {} clients: {}, samples: {}, min/max/mean: {}/{}/{:.2f}".format(
            split_name,
            len(sizes),
            int(sizes.sum()),
            int(sizes.min()),
            int(sizes.max()),
            float(sizes.mean()),
        )
    )


def _save_pickle(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as outfile:
        pickle.dump(payload, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print(">>> Saved: {}".format(path))


def main():
    print(">>> Loading Fashion-MNIST.")
    trainset = torchvision.datasets.FashionMNIST(
        root=DATASET_DIR, train=True, download=True
    )
    testset = torchvision.datasets.FashionMNIST(
        root=DATASET_DIR, train=False, download=True
    )

    train_images = _as_numpy(trainset.data)
    train_labels = _as_numpy(trainset.targets).astype(np.int64, copy=False)
    test_images = _as_numpy(testset.data)
    test_labels = _as_numpy(testset.targets).astype(np.int64, copy=False)

    rng = np.random.RandomState(SEED)
    train_x, train_y = _iid_partition(train_images, train_labels, NUM_USER, rng)
    test_x, test_y = _iid_partition(test_images, test_labels, NUM_USER, rng)

    train_data = _build_payload(train_x, train_y)
    test_data = _build_payload(test_x, test_y)

    _validate_partition(train_data, len(train_labels), "Train")
    _validate_partition(test_data, len(test_labels), "Test")

    if SAVE:
        _save_pickle(
            train_data, os.path.join(CPATH, "data", "train", TRAIN_FILENAME)
        )
        _save_pickle(test_data, os.path.join(CPATH, "data", "test", TEST_FILENAME))


if __name__ == "__main__":
    main()
