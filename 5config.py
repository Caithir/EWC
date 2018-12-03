import os
import torch
from collections import namedtuple
# 'D:', 'OneDrive - Duke University', 'research', 'EWC'
#BASE_LOG_DIR = ['logs']
#BASE_DATA_DIR = ['data']
#BASE_MODEL_DIR = ['models']

BASE_MODEL_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'models']
BASE_LOG_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'logs']
BASE_DATA_DIR = ['/usr', 'project', 'xtmp', 'EWC', 'EWC', 'data']

run_settings = {
    'print_freq': 20,
    # options are standard: no fisher stuff, fisher: only fisher stuff
    # 'experiments': ['standard', 'fisher'],
    #'experiments': ['standard'],
    'experiments': ['fisher'],
    # 'fisher_base_model': 'MLP3_cl-8.pth',
    'fisher_base_model': 'resnet18_cl-8.pth',
    'experiment_name': "5results"
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



fisher_hyperparams = {
        'lr': 1e-3,
        'start_epoch': 0,
        'epochs': 2,
        'momentum': .9,
        'weight_decay': 0,
        'arch': 'MLP',
        'batch_size': 64,
        'classes': list(range(5)),
        'grad_clip': 1,
        'lam': 15,
}


params = {}
params.update(norm_hyperparams)
# params.update(fisher_hyperparams)
params.update(system_settings)
params.update(run_settings)
params['relevant_params'] = {
        'lr': params['lr'],
	'ex': params['experiments'][0][:2],
        'arc': params['arch'],
        'gc': params['grad_clip'],
        'cl': len(params['classes']),
	'lam': params['lam'],
    }
config_items = params.keys()

config_gen = namedtuple('Config', config_items)

# config is name tuple ...
config = config_gen(**params)


def restore_config_from_dict(config_old):
    old_conf_gen = namedtuple('Config_old', config_old.keys())
    return old_conf_gen(**config_old)

def swap_config():
    params = {}
    params.update(fisher_hyperparams)
    params.update(system_settings)
    params.update(run_settings)

    global config
    config = config_gen(**params)
