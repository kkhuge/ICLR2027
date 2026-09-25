import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['mkdir', 'read_data_Mnist', 'read_data_Cifar10', 'Metrics', "MiniDataset", 'read_data_tinyimagenet']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def _filter_data_files(files, key):
    if key is None:
        return files

    key = str(key)
    exact_name = key if key.endswith('.pkl') else key + '.pkl'
    exact_files = [f for f in files if f == exact_name]
    if exact_files:
        return exact_files

    return [f for f in files if _matches_data_file(f, key)]


def _matches_data_file(filename, key):
    stem, ext = os.path.splitext(filename)
    if ext != '.pkl':
        return False

    key = str(key)
    if key.endswith('.pkl'):
        key = key[:-4]

    return (
        stem == key
        or stem.startswith(key + '_')
        or stem.endswith('_' + key)
        or '_' + key + '_' in stem
    )


def read_data_Mnist(train_data_dir, test_data_dir, key=None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    train_files = _filter_data_files(train_files, key)

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    test_files = _filter_data_files(test_files, key)

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.2860,), (0.3530,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target

def read_data_Cifar10(train_data_dir, test_data_dir, data_enhancement, key=None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    train_files = _filter_data_files(train_files, key)

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        if data_enhancement:
            train_data[cid] = Cifar10Dataset_train(v['x'], v['y'])
        else:
            train_data[cid] = Cifar10Dataset_test(v['x'], v['y'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    test_files = _filter_data_files(test_files, key)

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = Cifar10Dataset_test(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_data_Cifar100(train_data_dir, test_data_dir, data_enhancement, key=None):

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    # ----- read train -----
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    train_files = _filter_data_files(train_files, key)

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        if data_enhancement:
            train_data[cid] = Cifar100Dataset_train(v['x'], v['y'])
        else:
            train_data[cid] = Cifar100Dataset_test(v['x'], v['y'])

    # ----- read test -----
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    test_files = _filter_data_files(test_files, key)

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = Cifar100Dataset_test(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data

def read_data_tinyimagenet(train_data_dir, test_data_dir, data_enhancement, key=None):
    """Parses data in given train and test data directories for Tiny-ImageNet"""

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    # ----- read train -----
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    train_files = _filter_data_files(train_files, key)

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        if data_enhancement:
            train_data[cid] = TinyImageNetDataset_train(v['x'], v['y'])
        else:
            train_data[cid] = TinyImageNetDataset_test(v['x'], v['y'])

    # ----- read test -----
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    test_files = _filter_data_files(test_files, key)

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = TinyImageNetDataset_test(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data






class Cifar10Dataset_train(Dataset):
    def __init__(self, data, labels):
        super(Cifar10Dataset_train, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        # Ensure data is in uint8 format and has the correct shape for CIFAR-10
        self.data = self.data.astype("uint8")  # CIFAR-10 images are uint8
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        # Convert to PIL Image for transformation
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target

class Cifar10Dataset_test(Dataset):
    def __init__(self, data, labels):
        super(Cifar10Dataset_test, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        # Ensure data is in uint8 format and has the correct shape for CIFAR-10
        self.data = self.data.astype("uint8")  # CIFAR-10 images are uint8
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        # Convert to PIL Image for transformation
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target



class Cifar100Dataset_train(Dataset):
    def __init__(self, data, labels):
        super(Cifar100Dataset_train, self).__init__()
        self.data = np.array(data).astype("uint8")
        self.labels = np.array(labels).astype("int64")

        # CIFAR-100 normalization
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        img = self.transform(img)
        return img, self.labels[index]


class Cifar100Dataset_test(Dataset):
    def __init__(self, data, labels):
        super(Cifar100Dataset_test, self).__init__()
        self.data = np.array(data).astype("uint8")
        self.labels = np.array(labels).astype("int64")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        img = self.transform(img)
        return img, self.labels[index]


class TinyImageNetDataset_train(Dataset):
    def __init__(self, data, labels):
        super(TinyImageNetDataset_train, self).__init__()
        # Change 1: data is now a path list (List[str]) and cannot be cast to uint8.
        # Store the list directly, or convert it to a NumPy array without a dtype.
        self.image_paths = data
        self.labels = np.array(labels).astype("int64")

        # Tiny-ImageNet normalization
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Change 2: load images from their paths.
        path = self.image_paths[index]

        # Convert to RGB because some images may be grayscale (L) or RGBA.
        # Otherwise, the subsequent transform may fail.
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {path}, {e}")
            # If loading fails, optionally return a black image to avoid interrupting training.
            img = Image.new('RGB', (64, 64))

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]


class TinyImageNetDataset_test(Dataset):
    def __init__(self, data, labels):
        super(TinyImageNetDataset_test, self).__init__()
        # Change 1: store paths only here as well.
        self.image_paths = data
        self.labels = np.array(labels).astype("int64")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Change 2: load images from their paths.
        path = self.image_paths[index]

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {path}, {e}")
            img = Image.new('RGB', (64, 64))

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]


class Metrics(object):
    def __init__(self, clients, options, name=''):
        self.options = options
        num_rounds = options['num_round'] + 1
        self.bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.client_computations = {c.cid: [0] * num_rounds for c in clients}
        self.bytes_read = {c.cid: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.gradnorm_on_train_data = [0] * num_rounds
        self.graddiff_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,
                                                    options['seed'],
                                                    options['lr'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    'w' if options['noaverage'] else 'a')

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        if options['dis']:
            suffix = options['dis']
            self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
        self.graddiff_on_train_data[round_i] = train_stats['graddiff']

        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
        self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_every'] = self.options['eval_every']
        metrics['lr'] = self.options['lr']
        metrics['lr_schedule'] = self.options.get('lr_schedule', 'constant')
        metrics['num_epoch'] = self.options['num_epoch']
        metrics['batch_size'] = self.options['batch_size']
        for key in (
            'target_epsilon',
            'actual_epsilon',
            'dp_delta',
            'dp_noise_multiplier',
            'dp_clip_norm',
        ):
            if key in self.options:
                metrics[key] = self.options[key]

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['gradnorm_on_train_data'] = self.gradnorm_on_train_data
        metrics['graddiff_on_train_data'] = self.graddiff_on_train_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(str(metrics), ouf)
