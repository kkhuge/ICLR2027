import numpy as np
import torch.nn as nn
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch
import copy
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """

    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.initial_lr = options['lr']
        self.lr_schedule = options.get('lr_schedule', 'constant')
        self.lr_decay_exponent = float(
            options.get('lr_decay_exponent', 1.0)
        )
        if self.lr_decay_exponent < 0:
            raise ValueError('lr_decay_exponent must be non-negative.')
        if self.lr_schedule not in {
            'constant', 'inverse_time', 'inverse_sqrt', 'power'
        }:
            raise ValueError('Unsupported learning-rate schedule: {}'.format(
                self.lr_schedule
            ))
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options["model"] == '2nn' or options["model"] == 'linear' or options["model"] == "linear_regression" or \
                options["model"] == "2nnc":
            self.flat_data = True
        else:
            self.flat_data = False

        # # Setup local model and evaluate its statics
        # self.flops, self.params_num, self.model_bytes = \
        #     get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])
        self.flops = 1
        self.params_num = 1
        self.model_bytes = 1

    @property
    def model_bits(self):
        return self.model_bytes * 8

    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def set_learning_rate(self, round_i, local_step):
        """Set the learning rate for communication round ``round_i``."""
        if self.lr_schedule == 'constant':
            learning_rate = self.initial_lr
        elif self.lr_schedule == 'inverse_time':
            learning_rate = self.initial_lr / (round_i + 1)
        elif self.lr_schedule == 'inverse_sqrt':
            global_step = round_i * self.num_epoch + local_step
            learning_rate = self.initial_lr / math.sqrt(global_step + 1)
        else:
            learning_rate = self.initial_lr / (
                (round_i + 1) ** self.lr_decay_exponent
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        return learning_rate

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    # def local_train(self, train_dataloader, another_train_dataloader, round_i, global_c, **kwargs):
    #     """Train model locally and return new parameter and computation cost
    #
    #     Args:
    #         train_dataloader: DataLoader class in Pytorch
    #
    #     Returns
    #         1. local_solution: updated new parameter
    #         2. stat: Dict, contain stats
    #             2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
    #             2.2 loss
    #     """
    #     self.model.train()
    #     train_loss = train_acc = train_total = 0
    #     for epoch in range(self.num_epoch):
    #         train_loss = train_acc = train_total = 0
    #         for batch_idx, (x, y) in enumerate(train_dataloader):
    #             # from IPython import embed
    #             # embed()
    #             x = self.flatten_data(x)
    #             if self.gpu:
    #                 x, y = x.cuda(), y.cuda()
    #
    #             self.optimizer.zero_grad()
    #             pred = self.model(x)
    #
    #             # if torch.isnan(pred.max()):
    #             #     from IPython import embed
    #             #     embed()
    #
    #             loss = criterion(pred, y)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
    #             self.optimizer.step()
    #
    #             _, predicted = torch.max(pred, 1)
    #             correct = predicted.eq(y).sum().item()
    #             target_size = y.size(0)
    #
    #             train_loss += loss.item() * y.size(0)
    #             train_acc += correct
    #             train_total += target_size
    #
    #     local_solution = self.get_flat_model_params()
    #     param_dict = {"norm": torch.norm(local_solution).item(),
    #                   "max": local_solution.max().item(),
    #                   "min": local_solution.min().item()}
    #     comp = self.num_epoch * train_total * self.flops
    #     return_dict = {"comp": comp,
    #                    "loss": train_loss/train_total,
    #                    "acc": train_acc/train_total}
    #     return_dict.update(param_dict)
    #     return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class MSEWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(MSEWorker, self).__init__(
            model,
            optimizer,
            options
        )

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):

        self.model.train()
        train_loss = 0.
        train_total = 0
        train_acc = 0

        num_classes = 10  # Ten classes.

        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)

            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()

            pred = self.model(x)

            # 1. Obtain the class index y_label consistently.
            if y.dim() == 2 and y.size(1) == num_classes:
                y_label = y.argmax(dim=1).long()
            else:
                y_label = y.view(-1).long()

            # 2. Convert to one-hot encoding.
            target = F.one_hot(y_label, num_classes=num_classes).float()

            # 3. Compute standard MSE; this call returns only the loss variable.
            loss = F.mse_loss(pred, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                60
            )

            self.set_learning_rate(round_i, i)
            self.optimizer.step()

            target_size = y_label.size(0)

            predicted = torch.argmax(pred, dim=1)
            correct = predicted.eq(y_label.to(pred.device)).sum().item()

            train_acc += correct
            train_loss += loss.item() * target_size
            train_total += target_size

        local_solution = self.get_flat_model_params()

        param_dict = {
            "norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()
        }

        comp = self.num_epoch * train_total * self.flops

        return_dict = {
            "comp": comp,
            "loss": train_loss / train_total,
            "acc": train_acc / train_total
        }

        return_dict.update(param_dict)

        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):

        self.model.eval()
        test_acc = 0.
        test_loss = 0.
        test_total = 0

        num_classes = 10  # Ten classes.

        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)

                # 1. Obtain the class index y_label consistently.
                if y.dim() == 2 and y.size(1) == num_classes:
                    y_label = y.argmax(dim=1).long()
                else:
                    y_label = y.view(-1).long()

                # 2. Convert to one-hot encoding.
                target = F.one_hot(y_label, num_classes=num_classes).float()

                # 3. Compute standard MSE; this call also returns only the loss variable.
                loss = F.mse_loss(pred, target)

                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                predicted = torch.argmax(pred, dim=1)
                correct = predicted.eq(y_label.to(pred.device)).sum().item()

                test_acc += correct

        return test_acc, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

            self.set_learning_rate(round_i, i)
            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_grad(self, dataloader):
        all_x = []
        all_y = []
        self.optimizer.zero_grad()
        loss = 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        pred = self.model(all_x)
        loss = criterion(pred, all_y)
        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_jacobian(self, dataloader):
        self.optimizer.zero_grad()
        out_grad = []
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x).squeeze()
            for i in range(len(pred)):
                one_element_grad = []
                for j in range(len(pred[i])):
                    one_out_grad_flat = get_flat_grad(pred[i][j], self.model.parameters(), create_graph=True)
                    one_element_grad.append(one_out_grad_flat)
                one_element_grad = torch.hstack(one_element_grad)
                out_grad.append(one_element_grad)
        out_grad = torch.vstack(out_grad)

        return out_grad

    def get_error(self, test_dataloader):
        error = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                true_value = np.zeros((len(y), 10))
                true_value[np.arange(len(y)), y] = 1

                pred = self.model(x)
                error = error + np.linalg.norm(pred - true_value, ord='fro')
                print(pred)
                print(true_value)
        return error

    # Return decoupled backbone and head gradients.
    def get_separated_grads(self, dataloader):
        """
        Return decoupled backbone and head gradients.
        Accumulate gradients to avoid exhausting memory.
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_samples = 0

        # 1. Iterate over local data and accumulate gradients.
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            pred = self.model(x)
            loss = criterion(pred, y)

            # Multiply loss by batch size to obtain the exact dataset-wide mean gradient later.
            batch_size = y.size(0)
            (loss * batch_size).backward()
            total_samples += batch_size

        # 2. Separate and flatten backbone and head gradients.
        grad_b = []
        grad_h = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Divide by the total sample count to obtain the mean gradient.
                    avg_grad = param.grad.detach().clone() / total_samples
                    if "readout" in name:
                        grad_h.append(avg_grad.view(-1))
                    else:
                        grad_b.append(avg_grad.view(-1))

        # Clear gradients so subsequent training is unaffected.
        self.optimizer.zero_grad()

        # Concatenate into one-dimensional vectors.
        grad_b_flat = torch.cat(grad_b) if len(grad_b) > 0 else torch.tensor([])
        grad_h_flat = torch.cat(grad_h) if len(grad_h) > 0 else torch.tensor([])

        return grad_b_flat, grad_h_flat


