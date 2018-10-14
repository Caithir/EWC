import os
import torch
from collections import namedtuple
# 'D:', 'OneDrive - Duke University', 'research', 'EWC'
# BASE_LOG_DIR = ['logs']
# BASE_DATA_DIR = ['data']
# BASE_MODEL_DIR = ['models']

BASE_MODEL_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'models']
BASE_LOG_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'logs']
BASE_DATA_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'data']

run_settings = {
    'print_freq': 100,
    # options are standard: no fisher stuff, fisher: only fisher stuff
    # 'experiments': ['standard', 'fisher'],
    # 'experiments': ['standard'],
    'experiments': ['fisher'],
    'fisher_base_model': 'lr-0.01_arc-resnet18_gc-5_cl-8.pth',
    'experiment_name': "testrunss"
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


hyperparams = {
        'lr': 1e-2,
        'start_epoch': 0,
        'epochs': 2,
        'momentum': .9,
        'weight_decay': 0,
        'arch': 'resnet18',
        'batch_size': 64,
        'classes': list(range(8)),
        'grad_clip': 5
}


params = {}
params.update(hyperparams)
params.update(system_settings)
params.update(run_settings)

config_items = params.keys()

config_gen = namedtuple('Config', config_items)

# config is name tuple ...
config = config_gen(**params)


def restore_config_from_dict(config):
    return config_gen(**config)

