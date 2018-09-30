from torch.nn.utils.convert_parameters import parameters_to_vector

import torch
from Utils import get_model_from_config
from config import restore_config_from_dict
import matplotlib.pyplot as plt


def fisher_graph(filename):

    checkpoint = torch.load(filename)
    fisher_diag = checkpoint['FI']
    config = restore_config_from_dict(checkpoint['confid'])
    model = get_model_from_config(config)

    ordered_params = [fisher_diag[name] for name, _ in model.named_parameters()]
    fisher_vector = parameters_to_vector(ordered_params)


    plt.plot(fisher_vector)