"""FedAvg with explicit two-level federated-generalization evaluation."""

import csv
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.trainers.fedavg5 import DP_EPS, FedAvg5Trainer
from src.utils.worker_utils import Cifar10Dataset_test, MiniDataset


HIERGEN_DIR = os.path.join("result_hiergen", "fedavg5")


class FedAvgHierGenTrainer(FedAvg5Trainer):
    """Record within- and unseen-client generalization with explicit weighting.

    ``loss_list_train`` is the empirical loss on participating-client training
    data and ``loss_list_test`` is its independent 20% client-local validation
    loss.  This class additionally evaluates the same global model on the
        clients withheld before training.  Each quantity is averaged across
    clients, rather than across samples, to estimate the two expectations in
    the hierarchical FL definition.
    """

    def __init__(self, options, dataset, another_dataset):
        super(FedAvgHierGenTrainer, self).__init__(options, dataset, another_dataset)
        unseen_path = options.get("hiergen_unseen_file", "").strip()
        if not unseen_path:
            raise ValueError("--hiergen_unseen_file is required for fedavg_hiergen.")
        if not os.path.exists(unseen_path):
            raise FileNotFoundError("Unseen-client split does not exist: {}".format(unseen_path))

        with open(unseen_path, "rb") as infile:
            payload = pickle.load(infile)
        self.unseen_client_ids = list(payload["users"])
        dataset_name = str(options.get("dataset", ""))
        unseen_dataset_class = (
            MiniDataset
            if dataset_name.startswith(("mnist_", "emnist_", "femnist_", "fmnist_"))
            else Cifar10Dataset_test
        )
        self.unseen_client_data = [
            unseen_dataset_class(
                payload["user_data"][user]["x"], payload["user_data"][user]["y"]
            )
            for user in self.unseen_client_ids
        ]
        if not self.unseen_client_data:
            raise ValueError("The unseen-client split must contain at least one client.")
        self.loss_list_unseen = []
        self.acc_list_unseen = []
        self.hiergen_setting = options.get("hiergen_setting", "").strip()
        if not self.hiergen_setting:
            if "natural_niid" in dataset_name:
                self.hiergen_setting = "natural_niid"
            elif "iid" in dataset_name:
                self.hiergen_setting = "iid"
            else:
                self.hiergen_setting = "unknown"
        self.seed = int(options.get("seed", 0))
        legacy_weighting = options.get(
            "hiergen_client_weighting", "equal"
        ).strip().lower()
        self.metric_weighting = (
            options.get("hiergen_metric_weighting", "").strip().lower()
            or legacy_weighting
        )
        self.update_weighting = (
            options.get("hiergen_update_weighting", "").strip().lower()
            or legacy_weighting
        )
        if self.metric_weighting not in ("equal", "samples"):
            raise ValueError("hiergen client weighting must be equal or samples")
        if self.update_weighting not in ("equal", "samples"):
            raise ValueError("hiergen update weighting must be equal or samples")
        if self.update_weighting == "samples" and self.dp_noise_multiplier != 0:
            raise ValueError(
                "sample-weighted update aggregation is enabled only for sigma=0; "
                "a weighted client-DP mechanism requires a separate sensitivity analysis"
            )
        # Preserve the old CSV column while recording the two choices explicitly.
        self.client_weighting = self.metric_weighting
        print(
            ">>> HierGen weighting: metrics={} / updates={}".format(
                self.metric_weighting, self.update_weighting
            )
        )
        os.makedirs(HIERGEN_DIR, exist_ok=True)

    def _client_equal_local_test(self, use_eval_data):
        self.worker.set_flat_model_params(self.latest_model)
        losses = []
        accuracies = []
        for client in self.clients:
            correct, samples, loss_sum = client.local_test(use_eval_data=use_eval_data)
            if samples:
                losses.append(float(loss_sum) / samples)
                accuracies.append(float(correct) / samples)
        if not losses:
            raise RuntimeError("No nonempty participating-client datasets to evaluate.")
        return float(np.mean(losses)), float(np.mean(accuracies))

    def _sample_weighted_local_test(self, use_eval_data):
        self.worker.set_flat_model_params(self.latest_model)
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        for client in self.clients:
            correct, samples, loss_sum = client.local_test(
                use_eval_data=use_eval_data
            )
            total_correct += int(correct)
            total_samples += int(samples)
            total_loss += float(loss_sum)
        if total_samples <= 0:
            raise RuntimeError("No participating-client samples to evaluate.")
        return total_loss / total_samples, total_correct / total_samples

    def _unseen_client_test(self):
        self.worker.set_flat_model_params(self.latest_model)
        losses = []
        accuracies = []
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        for dataset in self.unseen_client_data:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            correct, loss_sum = self.worker.local_test(dataloader, dataloader)
            samples = len(dataset)
            if samples:
                losses.append(float(loss_sum) / samples)
                accuracies.append(float(correct) / samples)
                total_correct += int(correct)
                total_samples += int(samples)
                total_loss += float(loss_sum)
        if self.metric_weighting == "samples":
            if total_samples <= 0:
                raise RuntimeError("No unseen-client samples to evaluate.")
            return total_loss / total_samples, total_correct / total_samples
        return float(np.mean(losses)), float(np.mean(accuracies))

    def _participating_test(self, use_eval_data):
        if self.metric_weighting == "samples":
            return self._sample_weighted_local_test(use_eval_data)
        return self._client_equal_local_test(use_eval_data)

    def aggregate(self, solns):
        if self.update_weighting != "samples":
            return super(FedAvgHierGenTrainer, self).aggregate(solns)

        total_samples = float(sum(num_samples for num_samples, _ in solns))
        if total_samples <= 0:
            raise RuntimeError("Cannot aggregate an empty client sample set.")
        weighted_update = torch.zeros_like(self.latest_model)
        for num_samples, local_solution in solns:
            update = local_solution - self.latest_model
            update_norm = torch.norm(update)
            clip_scale = torch.clamp(
                self.dp_clip_norm / (update_norm + DP_EPS), max=1.0
            )
            weighted_update += (
                float(num_samples) / total_samples
            ) * update * clip_scale
        return (self.latest_model + self.server_lr * weighted_update).detach()

    def test_latest_model_on_traindata(self, round_i):
        begin_time = time.time()
        loss, accuracy = self._participating_test(use_eval_data=False)
        if self.print_result:
            print(
                "= Participating train = round: {} / acc: {:.3%} / loss: {:.4f} / time: {:.2f}s"
                .format(round_i, accuracy, loss, time.time() - begin_time)
            )
        return loss, accuracy

    def test_latest_model_on_evaldata(self, round_i):
        begin_time = time.time()
        validation_loss, validation_accuracy = self._participating_test(
            use_eval_data=True
        )
        unseen_loss, unseen_accuracy = self._unseen_client_test()
        self.loss_list_unseen.append(unseen_loss)
        self.acc_list_unseen.append(unseen_accuracy)
        if self.print_result and round_i % self.eval_every == 0:
            print(
                "= HierGen = round: {} / val loss: {:.4f} / unseen loss: {:.4f} / time: {:.2f}s"
                .format(round_i, validation_loss, unseen_loss, time.time() - begin_time)
            )
        return validation_loss, validation_accuracy

    def train(self):
        super(FedAvgHierGenTrainer, self).train()
        train_loss = np.asarray(self.loss_list_train, dtype=float)
        validation_loss = np.asarray(self.loss_list_test, dtype=float)
        unseen_loss = np.asarray(self.loss_list_unseen, dtype=float)
        length = min(len(train_loss), len(validation_loss), len(unseen_loss))
        train_loss = train_loss[:length]
        validation_loss = validation_loss[:length]
        unseen_loss = unseen_loss[:length]

        outputs = {
            "loss_participating_train": train_loss,
            "loss_participating_val": validation_loss,
            "loss_unseen_clients": unseen_loss,
            "acc_participating_train": np.asarray(
                self.acc_list_train, dtype=float
            )[:length],
            "acc_participating_val": np.asarray(self.acc_list_test, dtype=float)[:length],
            "acc_unseen_clients": np.asarray(self.acc_list_unseen, dtype=float)[:length],
            "gap_within_client": validation_loss - train_loss,
            "gap_unseen_client": unseen_loss - validation_loss,
            "gap_total": unseen_loss - train_loss,
        }
        print(">>> Saved two-level generalization curves:")
        for name, values in outputs.items():
            path = os.path.join(HIERGEN_DIR, name + "_" + self.output_stem + ".npy")
            np.save(path, values)
            print("    {}".format(path))

        csv_path = os.path.join(
            HIERGEN_DIR, "curves_" + self.output_stem + ".csv"
        )
        fieldnames = [
            "round",
            "setting",
            "dp_level",
            "train_loss",
            "participating_val_loss",
            "unseen_loss",
            "gen_s",
            "gen_c",
            "gen_total",
            "train_accuracy",
            "participating_val_accuracy",
            "unseen_accuracy",
            "seed",
            "client_weighting",
            "metric_weighting",
            "update_weighting",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for checkpoint in range(length):
                writer.writerow({
                    "round": checkpoint,
                    "setting": self.hiergen_setting,
                    "dp_level": "client",
                    "train_loss": outputs["loss_participating_train"][checkpoint],
                    "participating_val_loss": outputs["loss_participating_val"][checkpoint],
                    "unseen_loss": outputs["loss_unseen_clients"][checkpoint],
                    "gen_s": outputs["gap_within_client"][checkpoint],
                    "gen_c": outputs["gap_unseen_client"][checkpoint],
                    "gen_total": outputs["gap_total"][checkpoint],
                    "train_accuracy": outputs["acc_participating_train"][checkpoint],
                    "participating_val_accuracy": outputs["acc_participating_val"][checkpoint],
                    "unseen_accuracy": outputs["acc_unseen_clients"][checkpoint],
                    "seed": self.seed,
                    "client_weighting": self.client_weighting,
                    "metric_weighting": self.metric_weighting,
                    "update_weighting": self.update_weighting,
                })
        print("    {}".format(csv_path))
