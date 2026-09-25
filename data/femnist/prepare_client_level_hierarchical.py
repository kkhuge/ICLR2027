"""Prepare paired FEMNIST writer-natural and IID-reshuffled splits.

The source files are the 62-class federated EMNIST HDF5 files published by
TensorFlow Federated.  For a fixed seed, the two settings use exactly the same
writers, samples, client-size vector, participating/unseen client counts, and
local 80/20 split rule.  Only the assignment of samples to client slots differs.
"""

import argparse
import json
import os
import pickle

import h5py
import numpy as np


VAL_FRACTION = 0.20


def _payload(users, user_data, metadata):
    return {
        "users": list(users),
        "user_data": user_data,
        "num_samples": [len(user_data[user]["y"]) for user in users],
        "hiergen_metadata": metadata,
    }


def _write_pickle(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as outfile:
        pickle.dump(value, outfile, pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _read_writer(train_examples, test_examples, writer_id):
    train_group = train_examples[writer_id]
    test_group = test_examples[writer_id]
    pixels = np.concatenate(
        (train_group["pixels"][()], test_group["pixels"][()]), axis=0
    )
    labels = np.concatenate(
        (train_group["label"][()], test_group["label"][()]), axis=0
    ).astype(np.int64, copy=False)

    # TFF FEMNIST uses 1 for background and 0 for ink.  The project's LeNet
    # grayscale pipeline expects uint8 images with bright ink on dark background.
    pixels = np.rint((1.0 - pixels) * 255.0)
    pixels = np.clip(pixels, 0.0, 255.0).astype(np.uint8)
    return pixels, labels


def _convert_pixels(pixels):
    pixels = np.rint((1.0 - pixels) * 255.0)
    return np.clip(pixels, 0.0, 255.0).astype(np.uint8)


def _read_writer_tff_split(train_examples, test_examples, writer_id):
    """Read one writer without changing TFF's original train/test split."""
    train_group = train_examples[writer_id]
    test_group = test_examples[writer_id]
    train_slot = (
        _convert_pixels(train_group["pixels"][()]),
        train_group["label"][()].astype(np.int64, copy=False),
    )
    test_slot = (
        _convert_pixels(test_group["pixels"][()]),
        test_group["label"][()].astype(np.int64, copy=False),
    )
    return train_slot, test_slot


def _split_participating(slots, seed):
    rng = np.random.RandomState(seed)
    train_data = {}
    validation_data = {}
    split_indices = {}
    for slot_id, (pixels, labels) in enumerate(slots):
        shuffled = rng.permutation(len(labels))
        validation_count = max(1, int(round(VAL_FRACTION * len(labels))))
        validation_count = min(validation_count, len(labels) - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]
        train_data[slot_id] = {
            "x": pixels[train_indices],
            "y": labels[train_indices],
        }
        validation_data[slot_id] = {
            "x": pixels[validation_indices],
            "y": labels[validation_indices],
        }
        split_indices[str(slot_id)] = {
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        }
    return train_data, validation_data, split_indices


def _unseen_payload(slots, first_slot):
    users = []
    user_data = {}
    for offset, (pixels, labels) in enumerate(slots):
        slot_id = first_slot + offset
        users.append(slot_id)
        user_data[slot_id] = {"x": pixels, "y": labels}
    return users, user_data


def _reshuffle_slots(slots, seed):
    """IID-reshuffle one TFF split while preserving its client-size vector."""
    sizes = [len(labels) for _, labels in slots]
    pixels = np.concatenate([images for images, _ in slots], axis=0)
    labels = np.concatenate([targets for _, targets in slots], axis=0)
    permutation = np.random.RandomState(seed).permutation(len(labels))
    shuffled = []
    cursor = 0
    for size in sizes:
        indices = permutation[cursor:cursor + size]
        shuffled.append((pixels[indices], labels[indices]))
        cursor += size
    assert cursor == len(labels)
    return shuffled


def _save_tff_split_setting(root, setting, seed, train_slots, validation_slots,
                            num_participating, base_metadata,
                            unseen_evaluation="all", key_override=None):
    """Save original TFF train/test and the selected unseen evaluation split."""
    key = key_override or "hiergen_client_{}_tffsplit_seed{}".format(
        setting, seed
    )
    participating_users = list(range(num_participating))
    train_data = {}
    validation_data = {}
    for client_id in participating_users:
        train_pixels, train_labels = train_slots[client_id]
        val_pixels, val_labels = validation_slots[client_id]
        train_data[client_id] = {"x": train_pixels, "y": train_labels}
        validation_data[client_id] = {"x": val_pixels, "y": val_labels}

    unseen_slots = []
    for train_slot, validation_slot in zip(
        train_slots[num_participating:], validation_slots[num_participating:]
    ):
        if unseen_evaluation == "test_only":
            unseen_slots.append(validation_slot)
        else:
            unseen_slots.append((
                np.concatenate((train_slot[0], validation_slot[0]), axis=0),
                np.concatenate((train_slot[1], validation_slot[1]), axis=0),
            ))
    unseen_users, unseen_data = _unseen_payload(
        unseen_slots, num_participating
    )
    metadata = dict(base_metadata)
    metadata.update({
        "setting": setting,
        "local_split": "original_tff_train_test",
        "unseen_evaluation": unseen_evaluation,
        "num_participating_clients": num_participating,
        "num_unseen_clients": len(unseen_users),
    })

    train_path = os.path.join(root, "data", "train", key + ".pkl")
    validation_path = os.path.join(root, "data", "test", key + ".pkl")
    unseen_path = os.path.join(root, "data", "hiergen", "unseen_" + key + ".pkl")
    _write_pickle(train_path, _payload(participating_users, train_data, metadata))
    _write_pickle(
        validation_path,
        _payload(participating_users, validation_data, metadata),
    )
    _write_pickle(unseen_path, _payload(unseen_users, unseen_data, metadata))
    return {
        "train_path": train_path,
        "validation_path": validation_path,
        "unseen_path": unseen_path,
        "train_samples": int(sum(len(v["y"]) for v in train_data.values())),
        "validation_samples": int(
            sum(len(v["y"]) for v in validation_data.values())
        ),
        "unseen_samples": int(sum(len(v["y"]) for v in unseen_data.values())),
    }


def _save_setting(root, setting, seed, participating_slots, unseen_slots,
                  base_metadata, split_seed):
    key = "hiergen_client_{}_seed{}".format(setting, seed)
    train_data, validation_data, split_indices = _split_participating(
        participating_slots, split_seed
    )
    participating_users = list(range(len(participating_slots)))
    unseen_users, unseen_data = _unseen_payload(
        unseen_slots, len(participating_slots)
    )
    metadata = dict(base_metadata)
    metadata.update({
        "setting": setting,
        "split_seed": int(split_seed),
        "validation_fraction": VAL_FRACTION,
        "num_participating_clients": len(participating_users),
        "num_unseen_clients": len(unseen_users),
    })

    train_path = os.path.join(root, "data", "train", key + ".pkl")
    validation_path = os.path.join(root, "data", "test", key + ".pkl")
    unseen_path = os.path.join(root, "data", "hiergen", "unseen_" + key + ".pkl")
    _write_pickle(train_path, _payload(participating_users, train_data, metadata))
    _write_pickle(
        validation_path,
        _payload(participating_users, validation_data, metadata),
    )
    _write_pickle(unseen_path, _payload(unseen_users, unseen_data, metadata))
    return {
        "train_path": train_path,
        "validation_path": validation_path,
        "unseen_path": unseen_path,
        "local_split_indices": split_indices,
        "train_samples": int(sum(len(v["y"]) for v in train_data.values())),
        "validation_samples": int(
            sum(len(v["y"]) for v in validation_data.values())
        ),
        "unseen_samples": int(sum(len(v["y"]) for v in unseen_data.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_participating", type=int, default=500)
    parser.add_argument("--num_unseen", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local_split",
        choices=("random_80_20", "tff_original"),
        default="random_80_20",
    )
    parser.add_argument(
        "--unseen_evaluation",
        choices=("all", "test_only"),
        default="all",
    )
    args = parser.parse_args()
    if args.num_participating <= 0 or args.num_unseen <= 0:
        raise ValueError("client counts must be positive")

    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, "raw")
    train_h5_path = os.path.join(raw_dir, "fed_emnist_train.h5")
    test_h5_path = os.path.join(raw_dir, "fed_emnist_test.h5")
    for path in (train_h5_path, test_h5_path):
        if not os.path.exists(path):
            raise FileNotFoundError("Missing official FEMNIST source file: " + path)

    num_total = args.num_participating + args.num_unseen
    selection_rng = np.random.RandomState(args.seed)
    with h5py.File(train_h5_path, "r") as train_h5, h5py.File(
        test_h5_path, "r"
    ) as test_h5:
        train_examples = train_h5["examples"]
        test_examples = test_h5["examples"]
        writer_ids = sorted(set(train_examples.keys()) & set(test_examples.keys()))
        if len(writer_ids) < num_total:
            raise RuntimeError(
                "Only {} eligible writers; need {}".format(len(writer_ids), num_total)
            )
        chosen_indices = selection_rng.permutation(len(writer_ids))[:num_total]
        selected_writer_ids = [writer_ids[int(index)] for index in chosen_indices]
        if args.local_split == "tff_original":
            original_splits = [
                _read_writer_tff_split(train_examples, test_examples, writer_id)
                for writer_id in selected_writer_ids
            ]
        else:
            natural_slots = [
                _read_writer(train_examples, test_examples, writer_id)
                for writer_id in selected_writer_ids
            ]

    if args.local_split == "tff_original":
        natural_train_slots = [pair[0] for pair in original_splits]
        natural_validation_slots = [pair[1] for pair in original_splits]
        iid_train_slots = _reshuffle_slots(natural_train_slots, args.seed + 10000)
        iid_validation_slots = _reshuffle_slots(
            natural_validation_slots, args.seed + 11000
        )
        base_metadata = {
            "dataset": "FEMNIST / Federated EMNIST (62 classes)",
            "source_url": (
                "https://storage.googleapis.com/tff-datasets-public/"
                "fed_emnist.tar.bz2"
            ),
            "source_train_h5": train_h5_path,
            "source_test_h5": test_h5_path,
            "partition_seed": int(args.seed),
            "selected_writer_ids": selected_writer_ids,
            "num_classes": 62,
            "pixel_conversion": "uint8(round((1 - tff_pixel) * 255))",
            "client_averaging": "samples",
        }
        natural_key = None
        iid_key = None
        if args.unseen_evaluation == "test_only":
            # Short aliases keep all Windows output paths comfortably below MAX_PATH.
            natural_key = "hn0"
            iid_key = "hi0"
        natural_result = _save_tff_split_setting(
            root, "natural_niid", args.seed,
            natural_train_slots, natural_validation_slots,
            args.num_participating, base_metadata,
            unseen_evaluation=args.unseen_evaluation,
            key_override=natural_key,
        )
        iid_result = _save_tff_split_setting(
            root, "iid", args.seed,
            iid_train_slots, iid_validation_slots,
            args.num_participating, base_metadata,
            unseen_evaluation=args.unseen_evaluation,
            key_override=iid_key,
        )
        audit = {
            "partition_seed": int(args.seed),
            "local_split": "original_tff_train_test",
            "selected_writer_ids": selected_writer_ids,
            "participating_writer_ids": selected_writer_ids[:args.num_participating],
            "unseen_writer_ids": selected_writer_ids[args.num_participating:],
            "natural_outputs": natural_result,
            "iid_outputs": iid_result,
        }
        audit_suffix = (
            "tffsplit_testonly" if args.unseen_evaluation == "test_only"
            else "tffsplit"
        )
        audit_path = os.path.join(
            root, "data", "hiergen",
            "partition_audit_client_level_{}_seed{}.json".format(
                audit_suffix, args.seed
            ),
        )
        with open(audit_path, "w", encoding="utf-8") as outfile:
            json.dump(audit, outfile, ensure_ascii=False, indent=2)
        print(
            "prepared original-TFF FEMNIST pair: participating={} unseen={} "
            "unseen_evaluation={}".format(
                args.num_participating, args.num_unseen, args.unseen_evaluation
            ),
            flush=True,
        )
        print("natural: {}".format(natural_result), flush=True)
        print("iid: {}".format(iid_result), flush=True)
        print("audit: {}".format(audit_path), flush=True)
        return

    slot_sizes = [len(labels) for _, labels in natural_slots]
    all_pixels = np.concatenate([pixels for pixels, _ in natural_slots], axis=0)
    all_labels = np.concatenate([labels for _, labels in natural_slots], axis=0)
    reshuffle_rng = np.random.RandomState(args.seed + 10000)
    global_permutation = reshuffle_rng.permutation(len(all_labels))
    iid_slots = []
    iid_assignment = {}
    cursor = 0
    for slot_id, slot_size in enumerate(slot_sizes):
        indices = global_permutation[cursor:cursor + slot_size]
        iid_slots.append((all_pixels[indices], all_labels[indices]))
        iid_assignment[str(slot_id)] = indices.tolist()
        cursor += slot_size
    assert cursor == len(all_labels)

    source_url = (
        "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"
    )
    base_metadata = {
        "dataset": "FEMNIST / Federated EMNIST (62 classes)",
        "source_url": source_url,
        "source_train_h5": train_h5_path,
        "source_test_h5": test_h5_path,
        "partition_seed": int(args.seed),
        "selected_writer_ids": selected_writer_ids,
        "slot_sizes": slot_sizes,
        "num_classes": 62,
        "pixel_conversion": "uint8(round((1 - tff_pixel) * 255))",
        "client_averaging": "equal",
    }
    participating_end = args.num_participating
    natural_result = _save_setting(
        root,
        "natural_niid",
        args.seed,
        natural_slots[:participating_end],
        natural_slots[participating_end:],
        base_metadata,
        args.seed + 20000,
    )
    iid_result = _save_setting(
        root,
        "iid",
        args.seed,
        iid_slots[:participating_end],
        iid_slots[participating_end:],
        base_metadata,
        args.seed + 30000,
    )

    audit = {
        "partition_seed": int(args.seed),
        "selected_writer_ids": selected_writer_ids,
        "participating_writer_ids": selected_writer_ids[:participating_end],
        "unseen_writer_ids": selected_writer_ids[participating_end:],
        "slot_sizes": slot_sizes,
        "iid_global_pool_assignment": iid_assignment,
        "natural_local_split_indices": natural_result.pop("local_split_indices"),
        "iid_local_split_indices": iid_result.pop("local_split_indices"),
        "natural_outputs": natural_result,
        "iid_outputs": iid_result,
    }
    audit_dir = os.path.join(root, "data", "hiergen")
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(
        audit_dir, "partition_audit_client_level_seed{}.json".format(args.seed)
    )
    with open(audit_path, "w", encoding="utf-8") as outfile:
        json.dump(audit, outfile, ensure_ascii=False, indent=2)

    print(
        "prepared FEMNIST client-level pair: participating={} unseen={} "
        "total_samples={}".format(
            args.num_participating, args.num_unseen, len(all_labels)
        ),
        flush=True,
    )
    print("natural: {}".format(natural_result), flush=True)
    print("iid: {}".format(iid_result), flush=True)
    print("audit: {}".format(audit_path), flush=True)


if __name__ == "__main__":
    main()
