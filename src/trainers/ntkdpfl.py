from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import Worker
from torch.optim import SGD

import numpy as np
import torch
import torch.nn.functional as F
import os
import math

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


loss_dir = "result_loss/ntkdpfl_ldp"
acc_dir = "result_acc/ntkdpfl_ldp"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)

if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


# ============================================================
# Hard-coded DP setting
# ============================================================

DP_CLIP_NORM = 0.2

# Change this for privacy-budget experiments: 0.5 / 1.0 / 2.0 / 4.0 / 6.0 / 8.0
TARGET_EPSILON = 2

# None means delta = 1 / num_clients
DELTA = None

# True: find sigma according to TARGET_EPSILON
# False: directly use DP_NOISE_MULTIPLIER
USE_TARGET_EPSILON = True

# Used only when USE_TARGET_EPSILON = False
DP_NOISE_MULTIPLIER = 1.0

SERVER_LR = 1.0

# Local-DP setting:
# Each selected client clips its own update and adds Gaussian noise before upload.
# For add/remove client adjacency, 1.0 matches clipped-update sensitivity C.
# For replacement client adjacency, set this to 2.0.
LDP_SENSITIVITY_MULTIPLIER = 1.0

# Smoothing perturbation scale. 1.0 means exactly match the effective
# LDP aggregation noise std tau_ldp = eta_s * sigma * C / sqrt(m).
# Larger values can be used for ablation only.
DP_GAUSSIAN_SMOOTH_SCALE = 1

DP_EPS = 1e-12


# ============================================================
# 10-class MSE setting
# ============================================================

NUM_CLASSES = 10


# ============================================================
# NTK-DPFL setting: decaying lambda
# ============================================================

# Actual NTK coefficient:
#     lambda_t = max(NTK_LAMBDA_MIN, NTK_LAMBDA0 * NTK_LAMBDA_DECAY ** round_i)
#
# Suggested for 20 local epochs:
#     NTK_LAMBDA0 = 1.0
#     NTK_LAMBDA_MIN = 0.125
#     NTK_LAMBDA_DECAY = 0.99
NTK_LAMBDA0 = 1
NTK_LAMBDA_MIN = 1
NTK_LAMBDA_DECAY = 0.99

# Parameter perturbation radius r = NTK_RHO * ||w||_2.
# Suggested grid: 1e-4, 3e-4, 1e-3, 3e-3.
NTK_RHO = 1e-3

# Add NTK smoothing every K local steps. 1 means every mini-batch.
NTK_REG_FREQ = 1

# Detach f_w(X) in ||f_{w+r u}(X)-f_w(X)||^2 for stability.
NTK_DETACH_BASE = True


# ============================================================
# Utilities
# ============================================================

def _tag_float(x):
    """Convert float to filename-safe string."""
    return str(x).replace(".", "p").replace("-", "m")


# ============================================================
# RDP accountant utilities
# We use RDP only as an accountant, and report standard
# client-level (epsilon, delta)-DP.
# ============================================================

def _logsumexp(log_terms):
    m = max(log_terms)
    if m == -float("inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in log_terms))


def _compute_rdp_sampled_gaussian(q, sigma, alpha):
    """
    Approximate RDP of Poisson-subsampled Gaussian mechanism
    for integer alpha.

    q: client sampling rate
    sigma: noise multiplier
    alpha: integer RDP order
    """
    if sigma <= 0:
        return float("inf")

    if q <= 0:
        return 0.0

    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)

    log_terms = []

    for i in range(alpha + 1):
        log_coef = (
            math.lgamma(alpha + 1)
            - math.lgamma(i + 1)
            - math.lgamma(alpha - i + 1)
        )

        if i == 0:
            log_q_term = (alpha - i) * math.log(1.0 - q)
        elif i == alpha:
            log_q_term = i * math.log(q)
        else:
            log_q_term = (
                i * math.log(q)
                + (alpha - i) * math.log(1.0 - q)
            )

        log_exp_term = (i * i - i) / (2.0 * sigma ** 2)

        log_terms.append(log_coef + log_q_term + log_exp_term)

    log_a = _logsumexp(log_terms)

    return log_a / (alpha - 1.0)


def compute_epsilon_from_sigma(q, sigma, steps, delta, orders=None):
    """
    Compose RDP over multiple communication rounds and convert
    to standard (epsilon, delta)-DP.
    """
    if orders is None:
        orders = list(range(2, 128))

    eps_list = []

    for alpha in orders:
        rdp_one_step = _compute_rdp_sampled_gaussian(q, sigma, alpha)
        rdp_total = steps * rdp_one_step
        eps = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)
        eps_list.append(eps)

    return min(eps_list)


def find_sigma_for_target_epsilon(target_epsilon, q, steps, delta):
    """
    Binary search noise multiplier sigma for target epsilon.
    """
    low = 0.05
    high = 50.0

    while compute_epsilon_from_sigma(q, high, steps, delta) > target_epsilon:
        high *= 2.0

    for _ in range(50):
        mid = (low + high) / 2.0
        eps_mid = compute_epsilon_from_sigma(q, mid, steps, delta)

        if eps_mid > target_epsilon:
            low = mid
        else:
            high = mid

    final_sigma = high
    final_epsilon = compute_epsilon_from_sigma(q, final_sigma, steps, delta)

    return final_sigma, final_epsilon


class NTKDPFLWorker(Worker):
    """
    Local worker for LDP-noise-aware Gaussian loss smoothing under 10-class MSE.

    Local objective:
        (1 - alpha_t) * Phi_i(theta)
        + alpha_t * Phi_i(theta + xi),

    where Phi_i is the 10-class MSE loss and
        xi ~ N(0, tau_ldp^2 I).

    Lambda is used as the clean/noisy loss mixing weight alpha_t.
    This class keeps the original trainer name for compatibility.
    """

    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options['num_epoch']
        super(NTKDPFLWorker, self).__init__(
            model,
            optimizer,
            optimizer_last_layer,
            options
        )

        # These values are set by NTKDPFLTrainer before each round.
        self.ntk_lambda_t = 0.0
        self.ntk_rho = NTK_RHO
        self.ntk_reg_freq = NTK_REG_FREQ
        self.ntk_detach_base = NTK_DETACH_BASE
        self.dp_perturb_std = 0.0
        self.loss_function = str(
            options.get("loss function", options.get("loss_function", "MSELoss"))
        ).lower()

    def _get_label(self, y, pred=None):
        """
        Convert y to class label [B] for 10-class classification.

        Supports:
            y: [B]
            y: [B, 1]
            y: one-hot [B, 10]
        """
        if y.dim() == 2 and y.size(1) == NUM_CLASSES:
            y_label = y.argmax(dim=1).long()
        else:
            y_label = y.view(-1).long()

        if pred is not None:
            y_min = y_label.min().item()
            y_max = y_label.max().item()
            num_classes = pred.size(1)

            if y_min < 0 or y_max >= num_classes:
                raise ValueError(
                    "Label out of range for 10-class MSE: "
                    f"label_min={y_min}, label_max={y_max}, "
                    f"num_classes={num_classes}, "
                    f"pred_shape={tuple(pred.shape)}, y_shape={tuple(y.shape)}. "
                    f"Labels must be in [0, {num_classes - 1}]."
                )

        return y_label

    def _mse_classification_loss(self, pred, y):
        """
        10-class MSE classification loss.

        pred: [B, 10]
        y:    [B] or [B, 1] or one-hot [B, 10]

        Loss:
            1/(2B) ||pred - one_hot(y)||_F^2
        """
        y = y.to(pred.device)

        if pred.dim() != 2:
            raise ValueError(
                f"Expected pred to be [B, {NUM_CLASSES}], "
                f"but got pred_shape={tuple(pred.shape)}."
            )

        if pred.size(1) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES}-class output, "
                f"but got pred_shape={tuple(pred.shape)}. "
                "If you use 10-class MSE, please set the model readout/output dimension to 10."
            )

        if y.dim() == 2 and y.size(1) == NUM_CLASSES:
            target = y.float()
        else:
            y_label = self._get_label(y, pred)
            target = F.one_hot(
                y_label.long(),
                num_classes=NUM_CLASSES
            ).float()

        target = target.to(pred.device)
        loss = F.mse_loss(pred, target)

        return loss

    def _ce_classification_loss(self, pred, y):
        """
        Cross-entropy classification loss.

        pred: [B, num_classes]
        y:    [B] or [B, 1] or one-hot [B, num_classes]

        Returns:
            loss, y_label
        """
        if pred.dim() != 2:
            raise ValueError(
                "Expected pred to be [B, num_classes], "
                f"but got pred_shape={tuple(pred.shape)}."
            )

        y_label = self._get_label(y, pred).to(pred.device)
        loss = F.cross_entropy(pred, y_label)

        return loss, y_label

    def _classification_loss(self, pred, y):
        """
        Dispatch classification loss according to options["loss function"].

        Supported:
            - MSELoss: original 10-class MSE behavior
            - CrossEntropyLoss / CE: standard cross-entropy loss
        """
        if (
            "cross" in self.loss_function
            or self.loss_function in {"ce", "celoss", "crossentropyloss"}
        ):
            return self._ce_classification_loss(pred, y)

        loss = self._mse_classification_loss(pred, y)
        y_label = self._get_label(y, pred).to(pred.device)

        return loss, y_label

    def _get_prediction(self, pred):
        """Convert 10-class model output to class label [B]."""
        if pred.dim() != 2 or pred.size(1) != NUM_CLASSES:
            raise ValueError(
                f"Expected pred to be [B, {NUM_CLASSES}], "
                f"but got pred_shape={tuple(pred.shape)}."
            )

        _, predicted = torch.max(pred, dim=1)
        return predicted



    def _loss_smoothing_loss(self, x, y, fallback_loss=None):
        """
        LDP-Gaussian loss smoothing without in-place parameter perturbation.

        This computes the task loss under a virtual Gaussian perturbed model:

            Phi_i(theta + xi),    xi ~ N(0, dp_perturb_std^2 I).

        The final local objective is formed in local_train as

            (1 - alpha_t) * Phi_i(theta)
            + alpha_t * Phi_i(theta + xi).

        Here dp_perturb_std is set by the trainer to match the effective
        LDP aggregation noise scale:
            tau_ldp = server_lr * sigma * C * sensitivity_multiplier / sqrt(m).
        """
        named_params = [
            (name, p) for name, p in self.model.named_parameters()
            if p.requires_grad
        ]

        if len(named_params) == 0:
            if fallback_loss is not None:
                return fallback_loss, 0.0
            return x.new_tensor(0.0), 0.0

        dp_std = float(getattr(self, "dp_perturb_std", 0.0))

        if dp_std <= 0.0:
            if fallback_loss is not None:
                return fallback_loss, 0.0
            return x.new_tensor(0.0), 0.0

        num_probes = 1
        smooth_loss = x.new_tensor(0.0)

        for _ in range(num_probes):
            noise_dict = {}

            for name, p in named_params:
                noise_dict[name] = torch.randn_like(p) * dp_std

            params_and_buffers = {}

            for name, p in self.model.named_parameters():
                if p.requires_grad and name in noise_dict:
                    params_and_buffers[name] = p + noise_dict[name]
                else:
                    params_and_buffers[name] = p

            # Clone buffers to avoid BatchNorm running stats being updated twice.
            for name, b in self.model.named_buffers():
                params_and_buffers[name] = b.detach().clone()

            pred_pert = functional_call(self.model, params_and_buffers, (x,))
            loss_pert, _ = self._classification_loss(pred_pert, y)
            smooth_loss = smooth_loss + loss_pert

        smooth_loss = smooth_loss / num_probes

        return smooth_loss, dp_std

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()

        train_loss = 0.0
        train_mse_loss = 0.0
        train_smooth_loss = 0.0
        train_total = 0
        train_acc = 0
        local_step = 0
        last_std = 0.0

        lambda_t = float(getattr(self, "ntk_lambda_t", 0.0))
        # Lambda is used as the smoothing mixing weight alpha_t in [0, 1].
        alpha_t = max(0.0, min(lambda_t, 1.0))
        reg_freq = int(getattr(self, "ntk_reg_freq", 1))

        for epoch in range(self.num_epoch):
            for x, y in train_dataloader:
                x = self.flatten_data(x)

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()

                pred = self.model(x)
                mse_loss, y_label = self._classification_loss(pred, y)

                use_smoothing = (
                    alpha_t > 0.0
                    and reg_freq > 0
                    and (local_step % reg_freq == 0)
                )

                if use_smoothing:
                    smooth_loss, last_std = self._loss_smoothing_loss(
                        x, y, fallback_loss=mse_loss
                    )
                    loss = (1.0 - alpha_t) * mse_loss + alpha_t * smooth_loss
                else:
                    smooth_loss = pred.new_tensor(0.0)
                    loss = mse_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                y_label = y_label.to(pred.device)
                predicted = self._get_prediction(pred)
                correct = predicted.eq(y_label).sum().item()
                batch_size = y_label.size(0)

                train_loss += loss.item() * batch_size
                train_mse_loss += mse_loss.item() * batch_size
                train_smooth_loss += smooth_loss.item() * batch_size
                train_acc += correct
                train_total += batch_size
                local_step += 1

        local_solution = self.get_flat_model_params()

        param_dict = {
            "norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()
        }

        comp = self.num_epoch * train_total * self.flops

        return_dict = {
            "comp": comp,
            "loss": train_loss / max(train_total, 1),
            "mse_loss": train_mse_loss / max(train_total, 1),
            # Backward-compatible key: ntk_loss now stores perturbed task loss.
            "ntk_loss": train_smooth_loss / max(train_total, 1),
            "smooth_loss": train_smooth_loss / max(train_total, 1),
            "acc": train_acc / max(train_total, 1),
            "ntk_lambda": alpha_t,
            "ntk_r": last_std,
            "dp_perturb_std": last_std,
        }

        return_dict.update(param_dict)

        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()

        test_acc = 0.0
        test_loss = 0.0
        test_total = 0

        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss, y_label = self._classification_loss(pred, y)

                y_label = y_label.to(pred.device)
                predicted = self._get_prediction(pred)
                correct = predicted.eq(y_label).sum().item()
                batch_size = y_label.size(0)

                test_loss += loss.item() * batch_size
                test_acc += correct
                test_total += batch_size

        return test_acc, test_loss


class NTKDPFLTrainer(BaseTrainer):
    """
    LDP-Gaussian loss smoothing under 10-class MSE loss.

    Privacy:
        Local-DP style aggregation. Each selected client clips and adds
        Gaussian noise before upload. The smoothing noise is internal
        training randomness and releases no additional statistics.
    """

    def __init__(self, options, dataset, another_dataset):
        self.error_train = []
        self.loss_list_train = []
        self.acc_list_train = []
        self.loss_list_test = []
        self.acc_list_test = []

        self.theta = []
        self.diff_nonlinear_linear = []
        self.weight_change = []

        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']

        self.optimizer = SGD(
            model.parameters(),
            lr=options['lr'],
            weight_decay=0.0001
        )

        self.optimizer_last_layer = SGD(
            model.readout.parameters(),
            lr=0.1,
            weight_decay=0.0001
        )

        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']

        worker = NTKDPFLWorker(
            model,
            self.optimizer,
            self.optimizer_last_layer,
            options
        )

        super(NTKDPFLTrainer, self).__init__(
            options,
            dataset,
            another_dataset,
            worker=worker
        )

        # ====================================================
        # Privacy budget setup
        # ====================================================

        self.dp_clip_norm = DP_CLIP_NORM
        self.server_lr = SERVER_LR

        self.num_clients = len(self.clients)
        self.sample_rate = self.clients_per_round / self.num_clients

        if DELTA is None:
            self.delta = 1.0 / self.num_clients
        else:
            self.delta = DELTA

        if USE_TARGET_EPSILON:
            self.target_epsilon = TARGET_EPSILON
            self.dp_noise_multiplier, self.actual_epsilon = (
                find_sigma_for_target_epsilon(
                    target_epsilon=self.target_epsilon,
                    q=self.sample_rate,
                    steps=self.num_round,
                    delta=self.delta
                )
            )
        else:
            self.target_epsilon = None
            self.dp_noise_multiplier = DP_NOISE_MULTIPLIER
            self.actual_epsilon = compute_epsilon_from_sigma(
                q=self.sample_rate,
                sigma=self.dp_noise_multiplier,
                steps=self.num_round,
                delta=self.delta
            )

        # Under local DP, each selected client adds independent noise with std
        # sigma * C before uploading. After averaging m clients, the effective
        # aggregate noise std is sigma * C / sqrt(m), then scaled by server_lr.
        self.local_dp_sensitivity_multiplier = LDP_SENSITIVITY_MULTIPLIER
        self.tau_dp = (
            self.server_lr
            * self.dp_noise_multiplier
            * self.dp_clip_norm
            / self.clients_per_round
        )

        # Gaussian smoothing noise used inside local training.
        self.dp_gaussian_smooth_scale = DP_GAUSSIAN_SMOOTH_SCALE
        self.dp_perturb_std = self.tau_dp * self.dp_gaussian_smooth_scale

        # Decaying NTK coefficient.
        self.ntk_lambda0 = NTK_LAMBDA0
        self.ntk_lambda_min = NTK_LAMBDA_MIN
        self.ntk_lambda_decay = NTK_LAMBDA_DECAY

        self.ntk_rho = NTK_RHO
        self.ntk_reg_freq = NTK_REG_FREQ

        eps_tag = (
            "eps{}".format(self.target_epsilon)
            if self.target_epsilon is not None
            else "sigma{}".format(self.dp_noise_multiplier)
        )

        self.result_tag = (
            "_" + eps_tag.replace(".", "p")
            + "_C{}".format(self.dp_clip_norm).replace(".", "p")
            + "_lam0{}".format(_tag_float(NTK_LAMBDA0))
            + "_lmin{}".format(_tag_float(NTK_LAMBDA_MIN))
            + "_decay{}".format(_tag_float(NTK_LAMBDA_DECAY))
            + "_ldp_gauss_smooth"
            + "_scale{}".format(_tag_float(DP_GAUSSIAN_SMOOTH_SCALE))
            + "_T{}".format(self.num_round)
        )

    def get_ntk_lambda(self, round_i):
        """
        Exponential decay:
            lambda_t = max(lambda_min, lambda0 * decay^t)
        """
        lambda_t = self.ntk_lambda0 * (self.ntk_lambda_decay ** round_i)
        lambda_t = max(self.ntk_lambda_min, lambda_t)
        return lambda_t

    def train(self):
        print('>>> Select {} clients per round\n'.format(
            self.clients_per_round
        ))

        print('>>> Local-DP Gaussian-Smoothing Privacy Setup:')
        print('    target epsilon     = {}'.format(self.target_epsilon))
        print('    actual epsilon     = {:.4f}'.format(self.actual_epsilon))
        print('    delta              = {:.6e}'.format(self.delta))
        print('    total clients      = {}'.format(self.num_clients))
        print('    clients per round  = {}'.format(self.clients_per_round))
        print('    sample rate q      = {:.6f}'.format(self.sample_rate))
        print('    rounds T           = {}'.format(self.num_round))
        print('    clip norm C        = {}'.format(self.dp_clip_norm))
        print('    noise multiplier σ = {:.6f}'.format(
            self.dp_noise_multiplier
        ))
        print('    LDP sensitivity mult = {}'.format(
            self.local_dp_sensitivity_multiplier
        ))
        print('    server lr          = {}'.format(self.server_lr))
        print('    tau_ldp            = {:.6e}'.format(self.tau_dp))
        print('    smooth scale       = {}'.format(self.dp_gaussian_smooth_scale))
        print('    dp perturb std     = {:.6e}'.format(self.dp_perturb_std))

        print('>>> LDP-noise-matched Gaussian Loss Smoothing Setup:')
        print('    NUM_CLASSES        = {}'.format(NUM_CLASSES))
        print('    NTK_LAMBDA0        = {}'.format(NTK_LAMBDA0))
        print('    NTK_LAMBDA_MIN     = {}'.format(NTK_LAMBDA_MIN))
        print('    NTK_LAMBDA_DECAY   = {}'.format(NTK_LAMBDA_DECAY))
        print('    DP_GAUSSIAN_SMOOTH_SCALE = {}'.format(DP_GAUSSIAN_SMOOTH_SCALE))
        print('    NTK_REG_FREQ       = {}'.format(self.ntk_reg_freq))
        print('    lambda is used as clean/noisy smoothing ratio alpha in [0, 1]\n')

        self.latest_model = self.worker.get_flat_model_params().detach().clone()

        for round_i in range(self.num_round):
            loss_test, accuracy_test = self.test_latest_model_on_evaldata(
                round_i
            )

            self.acc_list_test.append(accuracy_test)
            self.loss_list_test.append(loss_test)

            selected_clients = self.select_clients(seed=round_i)

            lambda_t = self.get_ntk_lambda(round_i)

            # Set round-wise NTK smoothing parameters on the shared worker.
            self.worker.ntk_lambda_t = lambda_t
            self.worker.ntk_rho = self.ntk_rho
            self.worker.ntk_reg_freq = self.ntk_reg_freq
            self.worker.ntk_detach_base = NTK_DETACH_BASE
            self.worker.dp_perturb_std = self.dp_perturb_std

            solns, stats = self.local_train(round_i, selected_clients)

            if round_i % 10 == 0:
                avg_mse = np.mean([s["mse_loss"] for s in stats])
                avg_smooth = np.mean([s.get("smooth_loss", s["ntk_loss"]) for s in stats])
                avg_lam = np.mean([s["ntk_lambda"] for s in stats])

                print(
                    "[LDP-GAUSS-SMOOTH DEBUG] round={} | alpha={:.6e} | "
                    "dp_std={:.6e} | mse={:.6e} | smooth={:.6e} | "
                    "mixed={:.6e} | ratio={:.6e}".format(
                        round_i,
                        avg_lam,
                        self.dp_perturb_std,
                        avg_mse,
                        avg_smooth,
                        (1.0 - avg_lam) * avg_mse + avg_lam * avg_smooth,
                        (avg_smooth) / (avg_mse + 1e-12)
                    )
                )

            self.metrics.extend_commu_stats(round_i, stats)

            self.latest_model = self.aggregate(solns)

        loss_test, accuracy_test = self.test_latest_model_on_evaldata(
            self.num_round
        )

        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)

        # np.save(
        #     loss_dir + '/loss_train_' + self.dataset + '_' + self.model ,
        #     self.loss_list_train
        # )
        #
        # np.save(
        #     acc_dir + '/acc_train_' + self.dataset + '_' + self.model ,
        #     self.acc_list_train
        # )

        # np.save(
        #     loss_dir + '/loss_test_' + self.dataset + '_' + self.model,
        #     self.loss_list_test
        # )

        np.save(
            acc_dir + '/acc_test_' + self.dataset + '_' + self.model,
            self.acc_list_test
        )

        self.metrics.write()

    def aggregate(self, solns):
        """
        Local-DP aggregation:

            1. each selected client computes its local update
            2. each client clips its own update to norm C
            3. each client adds Gaussian noise locally before upload
            4. server only averages privatized client updates
            5. server updates the global model

        The clipping/noising is implemented here for convenience, but it is
        mathematically equivalent to clients uploading clipped-and-noised
        updates under a local-DP style protocol.
        """
        privatized_update_sum = torch.zeros_like(self.latest_model)
        client_num = len(solns)

        noise_std_client = (
            self.dp_noise_multiplier
            * self.dp_clip_norm
            / math.sqrt(client_num)
        )

        for _, local_solution in solns:
            update = local_solution - self.latest_model
            update_norm = torch.norm(update)

            clip_scale = torch.clamp(
                self.dp_clip_norm / (update_norm + DP_EPS),
                max=1.0
            )

            clipped_update = update * clip_scale

            local_dp_noise = torch.randn_like(clipped_update) * noise_std_client
            privatized_update = clipped_update + local_dp_noise

            privatized_update_sum += privatized_update

        averaged_privatized_update = privatized_update_sum / client_num

        new_model = (
            self.latest_model
            + self.server_lr * averaged_privatized_update
        )

        return new_model.detach()
