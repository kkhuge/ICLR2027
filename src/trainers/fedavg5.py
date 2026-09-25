from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import MSEWorker, LrdWorker
from torch.optim import SGD

import numpy as np
import torch
import os
import math
import time


loss_dir = "result_loss/fedavg5"
acc_dir = "result_acc/fedavg5"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)

if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


# ============================================================
# Hard-coded DP setting
# ============================================================

DP_CLIP_NORM = 0.2

# Change this value as needed: 2.0 / 4.0 / 6.0 / 8.0.
TARGET_EPSILON = 20

# None means automatically setting it to 1 / num_clients.
DELTA =  None

# True: infer sigma from TARGET_EPSILON.
# False: use DP_NOISE_MULTIPLIER directly.
USE_TARGET_EPSILON = False

# Effective when USE_TARGET_EPSILON = False.
DP_NOISE_MULTIPLIER = 1e-8

SERVER_LR = 1.0

# Aggregate-release sensitivity used by the privacy accountant.
# For add/remove client adjacency, 1.0 matches clipped-update sensitivity C.
# For replacement client adjacency, set this to 2.0.
LDP_SENSITIVITY_MULTIPLIER = 1.0

DP_EPS = 1e-12


def _tag_value(value):
    return ("{:g}".format(value)).replace("-", "m").replace(".", "p")


# ============================================================
# RDP accountant utilities
# Use an RDP accountant to compute epsilon for the sampled Gaussian mechanism.
# Report the result as standard (epsilon, delta)-DP.
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

    # no subsampling
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

    # ensure high is large enough
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


class FedAvg5Trainer(BaseTrainer):
    """
    FedAvg with clipped client updates and aggregate Gaussian noise.

    For historical reproducibility, independent Gaussian vectors are drawn
    inside the client loop and then averaged. Only the released aggregate
    distribution, not individual client messages, is accounted as client DP.
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
            weight_decay=0
        )

        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']

        loss_function = str(options.get("loss function", "MSELoss")).lower()

        if "cross" in loss_function or loss_function in {"ce", "celoss"}:
            worker = LrdWorker(
                model,
                self.optimizer,
                options
            )
        else:
            worker = MSEWorker(
                model,
                self.optimizer,
                options
            )

        super(FedAvg5Trainer, self).__init__(
            options,
            dataset,
            another_dataset,
            worker=worker
        )

        # ====================================================
        # Privacy budget setup
        # ====================================================

        configured_clip_norm = options.get('dp_clip_norm')
        self.dp_clip_norm = (
            DP_CLIP_NORM
            if configured_clip_norm is None
            else float(configured_clip_norm)
        )
        if self.dp_clip_norm <= 0:
            raise ValueError('dp_clip_norm must be positive.')
        self.server_lr = SERVER_LR
        self.progress_every = max(1, int(options.get('progress_every', 10)))

        self.num_clients = len(self.clients)
        self.sample_rate = self.clients_per_round / self.num_clients

        configured_delta = options.get('dp_delta')
        if configured_delta is not None:
            self.delta = float(configured_delta)
        elif DELTA is None:
            self.delta = 1.0 / self.num_clients
        else:
            self.delta = DELTA
        if not 0 < self.delta < 1:
            raise ValueError('dp_delta must lie strictly between 0 and 1.')

        configured_target_epsilon = options.get('target_epsilon')
        configured_noise_multiplier = options.get('dp_noise_multiplier')
        if (
            configured_target_epsilon is not None
            and configured_noise_multiplier is not None
        ):
            raise ValueError(
                'Specify either target_epsilon or dp_noise_multiplier, not both.'
            )

        # An explicit command-line sigma takes precedence over the repository
        # default that calibrates sigma from TARGET_EPSILON.  This keeps sweeps
        # reproducible even if USE_TARGET_EPSILON is enabled globally.
        use_target_epsilon = (
            configured_target_epsilon is not None
            or (USE_TARGET_EPSILON and configured_noise_multiplier is None)
        )

        if use_target_epsilon:
            self.target_epsilon = (
                TARGET_EPSILON
                if configured_target_epsilon is None
                else float(configured_target_epsilon)
            )
            if self.target_epsilon <= 0:
                raise ValueError('target_epsilon must be positive.')

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
            self.dp_noise_multiplier = (
                DP_NOISE_MULTIPLIER
                if configured_noise_multiplier is None
                else float(configured_noise_multiplier)
            )
            if self.dp_noise_multiplier < 0:
                raise ValueError('dp_noise_multiplier must be non-negative.')

            # sigma=0 is a useful non-private clipped-FedAvg baseline.  It has
            # no finite client-level DP guarantee, so record epsilon as infinity
            # rather than trying to assign it a privacy budget.
            if self.dp_noise_multiplier == 0:
                self.actual_epsilon = float('inf')
            else:
                self.actual_epsilon = compute_epsilon_from_sigma(
                    q=self.sample_rate,
                    sigma=self.dp_noise_multiplier,
                    steps=self.num_round,
                    delta=self.delta
                )

        # Algorithm 1 draws one Gaussian vector after averaging the clipped
        # client updates. Its coordinate-wise standard deviation is sigma*C/m.
        self.local_dp_sensitivity_multiplier = LDP_SENSITIVITY_MULTIPLIER
        self.tau_dp = (
            self.server_lr
            * self.dp_noise_multiplier
            * self.dp_clip_norm
            * self.local_dp_sensitivity_multiplier
            / self.clients_per_round
        )

        privacy_tag = (
            "eps{}".format(_tag_value(self.target_epsilon))
            if self.target_epsilon is not None
            else "sigma{}".format(_tag_value(self.dp_noise_multiplier))
        )
        self.result_tag = (
            "_" + privacy_tag
            + "_C{}".format(_tag_value(self.dp_clip_norm))
            + "_clientdp"
            + "_sens{}".format(_tag_value(self.local_dp_sensitivity_multiplier))
            + "_T{}".format(self.num_round)
        )
        self.output_stem = (
            self.dataset
            + "_" + self.model
            + self.result_tag
            + "_lr{}".format(_tag_value(options['lr']))
            + "_tau{}".format(options['num_epoch'])
            + "_bs{}".format(options['batch_size'])
            + "_{}".format(options.get('lr_schedule', 'constant'))
            + "_seed{}".format(options['seed'])
        )
        result_suffix = options.get('result_suffix', '').strip()
        if result_suffix:
            if not all(
                char.isalnum() or char in ('-', '_')
                for char in result_suffix
            ):
                raise ValueError(
                    'result_suffix may contain only letters, digits, hyphens, '
                    'and underscores.'
                )
            self.output_stem += "_" + result_suffix

        # Persist the calibrated privacy configuration in metrics.json.
        options['target_epsilon'] = self.target_epsilon
        options['actual_epsilon'] = self.actual_epsilon
        options['dp_delta'] = self.delta
        options['dp_noise_multiplier'] = self.dp_noise_multiplier
        options['dp_clip_norm'] = self.dp_clip_norm

    def train(self):
        print('>>> Select {} clients per round\n'.format(
            self.clients_per_round
        ))

        print('>>> Local-DP FedAvg Privacy Setup:')
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
        print('    tau_dp             = {:.6e}\n'.format(self.tau_dp))

        self.latest_model = self.worker.get_flat_model_params().detach().clone()
        train_begin = time.time()

        for round_i in range(self.num_round):
            if round_i % 1 ==0:
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                loss_train, accuracy_train = self.test_latest_model_on_traindata(round_i)


                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
                self.acc_list_train.append(accuracy_train)
                self.loss_list_train.append(loss_train)

            selected_clients = self.select_clients(seed=round_i)

            solns, stats = self.local_train(round_i, selected_clients)

            self.metrics.extend_commu_stats(round_i, stats)

            self.latest_model = self.aggregate(solns)

            completed_round = round_i + 1
            if (
                completed_round == 1
                or completed_round % self.progress_every == 0
                or completed_round == self.num_round
            ):
                gap = self.loss_list_test[-1] - self.loss_list_train[-1]
                print(
                    '[progress] round={}/{} test_acc={:.4f} '
                    'train_loss={:.6f} test_loss={:.6f} gap={:.6f} '
                    'elapsed={:.1f}s'.format(
                        completed_round,
                        self.num_round,
                        self.acc_list_test[-1],
                        self.loss_list_train[-1],
                        self.loss_list_test[-1],
                        gap,
                        time.time() - train_begin,
                    ),
                    flush=True,
                )

        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        loss_train, accuracy_train = self.test_latest_model_on_traindata(self.num_round)

        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)

        self.acc_list_train.append(accuracy_train)
        self.loss_list_train.append(loss_train)

        output_paths = {
            'loss_train': os.path.join(
                loss_dir, 'loss_train_' + self.output_stem + '.npy'
            ),
            'acc_train': os.path.join(
                acc_dir, 'acc_train_' + self.output_stem + '.npy'
            ),
            'loss_test': os.path.join(
                loss_dir, 'loss_test_' + self.output_stem + '.npy'
            ),
            'acc_test': os.path.join(
                acc_dir, 'acc_test_' + self.output_stem + '.npy'
            ),
        }
        np.save(output_paths['loss_train'], self.loss_list_train)
        np.save(output_paths['acc_train'], self.acc_list_train)
        np.save(output_paths['loss_test'], self.loss_list_test)
        np.save(output_paths['acc_test'], self.acc_list_test)

        print('>>> Saved experiment curves:')
        for output_path in output_paths.values():
            print('    {}'.format(output_path))

        self.metrics.write()

    # def aggregate(self, solns):
    #     averaged_solution = torch.zeros_like(self.latest_model)
    #     accum_sample_num = 0
    #     for num_sample, local_solution in solns:
    #         accum_sample_num += num_sample
    #         averaged_solution += num_sample * local_solution
    #     averaged_solution /= accum_sample_num
    #     return averaged_solution.detach()

    def aggregate(self, solns):
        """
        Historical figure implementation: clip each client update, draw an
        independent Gaussian vector for each selected client, then average.
        The aggregate noise has standard deviation sigma*C/m, the same
        distribution as drawing one Gaussian vector after aggregation. This
        placement preserves the original random-number trajectory of the
        released experiment scripts; it is not a claim of local DP.
        """
        privatized_update_sum = torch.zeros_like(self.latest_model)
        client_num = len(solns)
        noise_std_client = (
            self.dp_noise_multiplier * self.dp_clip_norm / math.sqrt(client_num)
        )

        for _, local_solution in solns:
            update = local_solution - self.latest_model

            #clip
            update_norm = torch.norm(update)
            clip_scale = torch.clamp(
                self.dp_clip_norm / (update_norm + DP_EPS),
                max=1.0
            )

            clipped_update = update * clip_scale
            local_dp_noise = torch.randn_like(clipped_update) * noise_std_client
            privatized_update_sum += clipped_update + local_dp_noise

        averaged_privatized_update = privatized_update_sum / client_num

        new_model = (
            self.latest_model
            + self.server_lr * averaged_privatized_update
        )

        return new_model.detach()
