
from collections import namedtuple

import torch

params = ["fisher",
          'lr',
          'data',
          'momentum',
          "weight_decay",
          "arch",
          "gpu",
          "distributed",
          "batch_size",
          "print_freq",
          "start_epoch",
          'epochs',
          ]

config_gen = namedtuple("Config", params)

hyperparams = {
        "fisher": False,
        'lr': 1e-2,
        'data': "./data",
        'start_epoch': 0,
        'epochs': 2,
        'momentum': .9,
        "weight_decay": 0,
        "arch": "resnet18",
        "gpu": torch.device(0),
        "distributed": False,
        "batch_size": 64,
        "print_freq": 100,
}

# config is name tuple ...
config = config_gen(**hyperparams)

def restore_config_from_dict(config):
    return config_gen(**config)

