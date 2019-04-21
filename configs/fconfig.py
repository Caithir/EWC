import os
import torch
from kaggleData.kaggleDataset import KaggleData
from collections import namedtuple
# 'D:', 'OneDrive - Duke University', 'research', 'EWC'
#BASE_LOG_DIR = ['logs']
#BASE_DATA_DIR = ['data']
#BASE_MODEL_DIR = ['models']

BASE_MODEL_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'models', 'FMNIST']
BASE_LOG_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'logs', 'FMNIST']
BASE_DATA_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'data']

run_settings = {
    'print_freq': 20,
    # options are standard: no fisher stuff, fisher: only fisher stuff
    'experiments': ['standard'],
    'experiment_name': 'fmnist',
    'dataset': KaggleData,
    'dataset_classes': 10,
    'test_train_split': .7,

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
    'gradient'
]


norm_hyperparams = {
        'lr': 1e-3,
        'start_epoch': 0,
        'epochs': 40,
        'momentum': .8,
        'weight_decay': 0,
        'arch': 'resnet50',
        'lr_sched_factor': .2,
        'batch_size': 32,
        'classes': list(range(10)),
        'grad_clip': 10,
        'lam': 1000,
}



params = {}
params.update(norm_hyperparams)
params.update(system_settings)
params.update(run_settings)
params['relevant_params'] = {
        'lr': params['lr'],
        'ex': params['experiments'][0][:2],
        'arc': params['arch'],
        }
config_items = params.keys()

config_gen = namedtuple('Config', config_items)

# config is name tuple ...
config = config_gen(**params)


def restore_config_from_dict(config_old):
    old_conf_gen = namedtuple('Config_old', config_old.keys())
    return old_conf_gen(**config_old)
