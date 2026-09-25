# GLOBAL PARAMETERS
DATASETS = ['sent140', 'nist', 'shakespeare',
            'mnist', 'synthetic', 'cifar10', 'fmnist', 'femnist',
            'cifar100', 'tinyimagenet', 'emnist']
TRAINERS = {'fedavg': 'FedAvgTrainer',
            'fedavg4': 'FedAvg4Trainer',
            'fedavg5': 'FedAvg5Trainer',
            'fedavg_hiergen': 'FedAvgHierGenTrainer',
            'fedavg6': 'FedAvg6Trainer',
            'fedavg9': 'FedAvg9Trainer',
            'fedavg10': 'FedAvg10Trainer',
            'scaffold': 'ScaffoldTrainer',
            'feddyn': 'FedDynTrainer',
            'boontk': 'BooNTKTrainer',
            'ccvr': 'CCVRTrainer',
            'fedetf': 'FedETFTrainer',
            'feduv': 'FedUVTrainer',
            'fedbabu': 'FedBabuTrainer',
            'fedrep': 'FedRepTrainer',
            'fedfrth-p': 'FedFRTHPTrainer',
            'fedsmoo': 'FedSMOOTrainer',
            'fedgf': 'FedGFTrainer',
            'fedsam': 'FedSAMTrainer',
            "ntkdpfl": "NTKDPFLTrainer",}
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist' or dataset == 'nist' or dataset == 'fmnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 10}
            elif model == 'linear_regression':
                return {'input_shape': 784, 'num_class': 1}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'emnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 47}
            elif model == 'linear_regression':
                return {'input_shape': 784, 'num_class': 1}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 47}
        elif dataset == 'femnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 62}
            elif model == 'linear_regression':
                return {'input_shape': 784, 'num_class': 1}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 62}
        elif dataset == 'cifar10':
            if model == '2nn' or model == 'linear_regression':
                return {'input_shape': 3072, 'num_class': 10}
            if model == '2nnc' or model == 'linear_regression':
                return {'input_shape': 3*32*32, 'num_class': 1}
            else:
                return {'input_shape': (3, 32, 32), 'num_class': 10}
        elif dataset == 'cifar100':
            if model == '2nn' or model == 'linear_regression':
                return {'input_shape': 3072, 'num_class': 100}
            if model == '2nnc' or model == 'linear_regression':
                return {'input_shape': 3 * 32 * 32, 'num_class': 1}
            else:
                return {'input_shape': (3, 32, 32), 'num_class': 100}
        elif dataset == 'tinyimagenet':
            if model == '2nn' or model == 'linear_regression':
                # 3*64*64 = 12288
                return {'input_shape': 12288, 'num_class': 200}
            if model == '2nnc' or model == 'linear_regression':
                return {'input_shape': 3 * 64 * 64, 'num_class': 1}
            else:
                return {'input_shape': (3, 64, 64), 'num_class': 200}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()
