import numpy as np
import argparse
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import random

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



from src.utils.worker_utils import read_data_Mnist, read_data_Cifar10, read_data_Cifar100, read_data_tinyimagenet
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg5')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='cifar10_all_data_1_random_iid')
    # mnist_all_data_1_dirichlet_niid_0.5; tinyimagenet_niid_0.1;cifar100_all_data_1_dirichlet_niid_0.1
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='resnet18') #,lenet
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        help='use gpu (default: True)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=300) # 100,500,550
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=50)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64) #64
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5) #5,10
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1) #0.1, 0.001, 0.0001, 0.002
    parser.add_argument('--target_epsilon',
                        help=('target client-level privacy epsilon; when set, '
                              'the trainer calibrates the Gaussian noise multiplier'),
                        type=float,
                        default=None)
    parser.add_argument('--dp_clip_norm',
                        help='client-update clipping norm C',
                        type=float,
                        default=None)
    parser.add_argument('--dp_delta',
                        help='privacy delta; defaults to 1 / number of clients',
                        type=float,
                        default=None)
    parser.add_argument('--dp_noise_multiplier',
                        help=('fixed Gaussian noise multiplier; used only when '
                              'target_epsilon is not supplied'),
                        type=float,
                        default=None)
    parser.add_argument('--progress_every',
                        help='print a compact progress line every N rounds',
                        type=int,
                        default=10)
    parser.add_argument('--hiergen_unseen_file',
                        help=('pickle containing held-out, unseen clients for '
                              'the two-level generalization evaluation'),
                        type=str,
                        default='')
    parser.add_argument('--hiergen_setting',
                        help='hierarchical split label saved in the structured results',
                        type=str,
                        choices=['', 'iid', 'natural_niid', 'rotated'],
                        default='')
    parser.add_argument('--hiergen_client_weighting',
                        help=('legacy joint weighting for hierarchical training '
                              'and evaluation'),
                        type=str,
                        choices=['equal', 'samples'],
                        default='equal')
    parser.add_argument('--hiergen_metric_weighting',
                        help=('weighting for train/validation/unseen metrics; '
                              'defaults to hiergen_client_weighting'),
                        type=str,
                        choices=['', 'equal', 'samples'],
                        default='')
    parser.add_argument('--hiergen_update_weighting',
                        help=('weighting for client model updates; defaults to '
                              'hiergen_client_weighting'),
                        type=str,
                        choices=['', 'equal', 'samples'],
                        default='')
    parser.add_argument('--result_suffix',
                        help=('optional suffix appended only to saved curve '
                              'filenames; does not affect training'),
                        type=str,
                        default='')
    parser.add_argument('--lr_schedule',
                        help=('local learning-rate schedule: constant, inverse_time '
                              '(eta0 / (t + 1), where t is the communication round), '
                              'inverse_sqrt, or power '
                              '(eta0 / (t + 1)^a);'),
                        type=str,
                        choices=['constant', 'inverse_time', 'inverse_sqrt', 'power'],
                        default='inverse_time')
    parser.add_argument('--lr_decay_exponent',
                        help=('power-law exponent a; used when '
                              '--lr_schedule=power'),
                        type=float,
                        default=1.0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--loss function',
                        help='CrossEntropyLoss or MSELoss;',
                        type=str,
                        default='CrossEntropyLoss')
    parser.add_argument('--psi',
                        help='required accuracy;',
                        type=str,
                        default=0.9)
    parser.add_argument('--another dataset',
                        help='name of another dataset;',
                        type=str,
                        default='mnist_all_data_1_random_iid')
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()


    # Set seeds
    # np.random.seed(1 + options['seed'])
    # torch.manual_seed(12 + options['seed'])
    # if options['gpu']:
    #     torch.cuda.manual_seed_all(123 + options['seed'])

    # ✅ -------------------------
    # Force all random seeds and deterministic CUDA algorithms.
    seed = 42 + options['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if options['gpu']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable cuDNN autotuning and enforce deterministic algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Enable fully deterministic mode when PyTorch >= 1.8.
    torch.use_deterministic_algorithms(True)
    # ✅ -------------------------

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # read another data
    idx = options['another dataset'].find("_")
    if idx != -1:
        another_dataset_name, another_sub_data = options['another dataset'][:idx], options['another dataset'][idx + 1:]
    else:
        another_dataset_name, another_sub_data = options['another dataset'], None
    assert another_dataset_name in DATASETS, "{} not in another dataset {}!".format(another_dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data, another_dataset_name, another_sub_data


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data, another_dataset_name, another_sub_dataset = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')
    another_train_path = os.path.join('./data', another_dataset_name, 'data', 'train')
    another_test_path = os.path.join('./data', another_dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    if dataset_name == 'cifar10':
        if options['model'] == 'linear_regression':
            all_data_info = read_data_Cifar10(train_path, test_path, 0, sub_data )
        else:
            all_data_info = read_data_Cifar10(train_path, test_path, 0, sub_data)
    elif dataset_name == 'cifar100':
        all_data_info = read_data_Cifar100(train_path, test_path, 1, sub_data)
    elif dataset_name == 'tinyimagenet':
        all_data_info = read_data_tinyimagenet(train_path, test_path, 1, sub_data)
    else:
        all_data_info = read_data_Mnist(train_path, test_path, sub_data)

    if another_dataset_name == 'cifar10':
        if options['model'] == 'linear_regression':
            another_all_data_info = read_data_Cifar10(another_train_path, another_test_path, 0, another_sub_dataset )
        else:
            another_all_data_info = read_data_Cifar10(another_train_path, another_test_path, 1, another_sub_dataset)
    else:
        another_all_data_info = read_data_Mnist(another_train_path, another_test_path, another_sub_dataset)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info, another_all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()

#120+20
