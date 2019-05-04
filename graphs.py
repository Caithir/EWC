from torch.nn.utils.convert_parameters import parameters_to_vector

import torch
import numpy as np
import pandas as pd
from Utils import get_model_from_config
from configs.config import restore_config_from_dict
import matplotlib.pyplot as plt
from configs.config import config
plt.style.use('seaborn-white')

def fisher_graph(filename):

    checkpoint = torch.load(filename, map_location=config.gpu)
    fisher_diag = checkpoint['FI']
    config = restore_config_from_dict(checkpoint['confid'])
    model = get_model_from_config(config)

    ordered_params = [fisher_diag[name] for name, _ in model.named_parameters()]
    fisher_vector = parameters_to_vector(ordered_params)


    plt.plot(fisher_vector)



class weighter:

    def __init__(self, alpha=1/8):
        self.avg = 0
        self.alpha = alpha

    def __call__(self, d):

        self.avg = self.avg * (1 - self.alpha) + float(d) * self.alpha
        return self.avg



BASE_LOC = "../ewcResults"
cur = "8_data"
def read_file(sub_folder, gc, lam):
    filename = f'{BASE_LOC}/{cur}/{sub_folder}/gc{gc}_lam{lam}.csv'
    return np.genfromtxt(filename, delimiter=',', usecols=(1, 2), skip_header=1, names='iter, val', converters={2: weighter()})


def read_file_no_lam(sub_folder, gc):
    filename = f'{BASE_LOC}/{cur}/{sub_folder}/gc-{gc}.csv'
    return np.genfromtxt(filename, delimiter=',', usecols=(1, 2), skip_header=1, names='iter, val', converters={2: weighter()})


def make_baseline():
    plt.figure(figsize=(8, 4))
    train = read_file("train_accuracy", -1, -1)
    fish = read_file("fisher_accuracy", -1, -1)
    plt.subplot(1, 2, 1)
    plt.plot('iter', 'val', data=train, label='New Task')
    plt.plot('iter', 'val', data=fish, label='Old Task')

    plt.xlabel("Mini Batches")
    plt.ylabel("Training Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    fish_loss = read_file("loss", -1, -1)
    plt.plot('iter', 'val', data=fish_loss, label='Loss')
    plt.xlabel("Mini Batches")
    plt.ylabel("Training Loss")
    plt.legend()

    plt.show()



def gc_plot():

    global cur
    cur = 'grad_clip_raw_data'

    def format(gc):
        if gc == 5:
            return 0.5
        return gc

    for gc in [5, 1, 10, 100]:

        dt = read_file_no_lam('train_accuracy', gc)
        df = read_file_no_lam('fisher_accuracy', gc)
        plt.subplot(2, 1, 1)
        plt.plot('iter', 'val', data=dt, label=f'clip of: {format(gc)}')
        plt.subplot(2, 1, 2)
        plt.plot('iter', 'val', data=df, label=f'clip of: {format(gc)}')

    plt.xlabel("Mini Batches")
    plt.ylabel("Training Accuracy")
    plt.subplot(2, 1, 1)
    plt.legend()
    plt.title("Insensitivity to clipping threshold")


    plt.show()

def make_main():

    # Resnet mnist
    train = read_file("train_accuracy", 1, 1000)
    fish = read_file("fisher_accuracy", 1, 1000)
    plt.subplot(1, 2, 1)

    plt.plot('iter', 'val', data=train, label='New Task')
    plt.plot('iter', 'val', data=fish, label='Old Task')
    plt.ylim(70, 105)

    plt.xlabel("Mini Batches")
    plt.ylabel("Training Accuracy")
    plt.title("Resnet MNIST")
    plt.legend()
    plt.grid()

    # Resnet fmnist
    global cur
    cur ='fmnist_data'
    train = read_file("train_accuracy", 1, 1000)
    fish = read_file("fisher_accuracy", 1, 1000)
    plt.subplot(1, 2, 2)
    plt.plot('iter', 'val', data=train, label='New Task')
    plt.plot('iter', 'val', data=fish, label='Old Task')
    plt.ylim(70, 105)

    plt.xlabel("Mini Batches")
    plt.ylabel("Training Accuracy")
    plt.title("Resnet Fashion MNIST")
    plt.legend()
    plt.grid()
    plt.show()

# make_main()
make_baseline()
# gc_plot()


def foo(data):

    alpha = 1/8
    avg = 0
    new_data = []
    for d in data:
        avg = avg*(1-alpha) + d*alpha



