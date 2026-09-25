import time
from torch.utils.data import DataLoader
import torch
import copy


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, train_data, test_data, another_train_data, another_test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker
        self.train_data = train_data
        self.test_data = test_data
        self.another_train_data = another_train_data
        self.another_test_data = another_test_data

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Use shuffle=False when measuring the kernel matrix.
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.another_train_dataloader = DataLoader(another_train_data, batch_size=batch_size, shuffle=True)  # Use shuffle=False when measuring the kernel matrix.
        self.another_test_dataloader = DataLoader(another_test_data, batch_size=batch_size, shuffle=True)

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """Get model gradient"""
        grad_in_tenser = self.worker.get_flat_grads(self.train_dataloader)
        return grad_in_tenser.cpu().detach().numpy()

    def get_grad(self):
        stats = {'id': self.cid}
        grad_in_tenser = self.worker.get_grad(self.train_dataloader)
        return (len(self.train_data), grad_in_tenser.cpu().detach().numpy()), stats

    def get_prediction(self):
        "Get prediction value"
        prediction = self.worker.get_prediction(self.train_dataloader)
        return prediction.cpu().detach().numpy()

    def get_jacobian(self):
        """Get model gradient"""
        jacobian_in_tenser = self.worker.get_jacobian(self.train_dataloader)
        return jacobian_in_tenser

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}
        grads = self.get_flat_grads()  # Return grad in numpy array

        return (len(self.train_data), grads), stats

    def solve_jacobian(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        jacobian = self.get_jacobian()  # Return grad in numpy array

        return (len(self.train_data), jacobian), stats

    # Return the decoupled gradient for this client.
    def get_separated_grads(self):
        """Return this client's decoupled gradient over its full local training set."""
        grad_b_tensor, grad_h_tensor = self.worker.get_separated_grads(self.train_dataloader)
        num_samples = len(self.train_data)

        return num_samples, grad_b_tensor.cpu().numpy(), grad_h_tensor.cpu().numpy()

    def local_train(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i,**kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats


    def local_test(self, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
            another_dataloader, another_dataset = self.another_test_dataloader, self.another_test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data
            another_dataloader, another_dataset = self.another_train_dataloader, self.another_train_data

        tot_correct, loss = self.worker.local_test(dataloader,another_dataloader)

        return tot_correct, len(dataset), loss


