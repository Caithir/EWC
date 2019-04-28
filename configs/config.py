import os
import torch
import torchvision.datasets as datasets
from collections import namedtuple
# 'D:', 'OneDrive - Duke University', 'research', 'EWC'
# BASE_LOG_DIR = ['logs']
# BASE_DATA_DIR = ['data']
# BASE_MODEL_DIR = ['models']

BASE_MODEL_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'models']
BASE_LOG_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'logs']
BASE_DATA_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'data']

run_settings = {
    'print_freq': 20,
    # options are standard: no fisher stuff, fisher: only fisher stuff
    # 'experiments': ['standard', 'fisher'],
    'experiments': ['standard'],
    # 'experiments': ['fisher'],
    'fisher_base_model': 'resnet18_cl-8.pth',
    #'experiment_name': 'fmnist',
    'experiment_name': '8-classic',
    'dataset': datasets.MNIST,
    #'dataset': datasets.CIFAR10,
    #'dataset': datasets.FashionMNIST,
    'EMNIST_split': 'digits',
    'dataset_classes': 10,

}

system_settings = {
        'gpu': torch.device(0),
        'distributed': False,
        'data': os.path.join(*BASE_DATA_DIR),
        'models': os.path.join(*BASE_MODEL_DIR),
        'logs': os.path.join(*BASE_LOG_DIR)
        }

log_items = [
    'prec1',
    'loss',
    'CE_loss',
    'fisher_loss',
    'gradient'
]


norm_hyperparams = {
        'lr': 1e-3,
        'start_epoch': 0,
        'epochs': 40,
        'momentum': .8,
        'weight_decay': 0,
        'arch': 'resnet18',
        'batch_size': 16,
        'classes': list(range(8)),
        'grad_clip': 1,
        'lam': 1000,
}

dataset_names = {
    datasets.FashionMNIST: 'F',
    datasets.EMNIST: 'E'+run_settings['EMNIST_split'][:1],
    datasets.MNIST: 'M',
    datasets.CIFAR10: "C"
}

dataset_standardize_params = {
    datasets.MNIST: [(0.1307,), (0.3081,)],
    datasets.FashionMNIST: [],
    datasets.CIFAR10: [],
}



params = {}
params.update(norm_hyperparams)
# params.update(fisher_hyperparams)
params.update(system_settings)
params.update(run_settings)
params['standardize_params'] = dataset_standardize_params[run_settings['dataset']]
params['relevant_params'] = {
        'lr': params['lr'],
        'ex': params['experiments'][0][:2],
        'arc': params['arch'],
        'gc': params['grad_clip'],
        'cl': len(params['classes']),
        'lam': params['lam'],
        'dataset': dataset_names[params['dataset']]
        }
config_items = params.keys()

config_gen = namedtuple('Config', config_items)

# config is name tuple ...
config = config_gen(**params)


def restore_config_from_dict(config_old):
    old_conf_gen = namedtuple('Config_old', config_old.keys())
    return old_conf_gen(**config_old)



