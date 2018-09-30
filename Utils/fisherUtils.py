import argparse
import copy
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .utils import get_model_from_config, get_filename_from_config, AverageMeter
from config import config, restore_config_from_dict

def calc_fisher_utils(model=None, filename=None):

    if not model:
        if not filename:
            raise ValueError("Must have either filename or model for fisher calc")
        checkpoint = torch.load(filename)
        config = restore_config_from_dict(checkpoint["config"])

        model = get_model_from_config(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config.gpu)

    print("model")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # not making any steps, using to clear gradients
    optimizer = torch.optim.SGD(model.parameters(), 0)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

    # train for one epoch
    fisher_diag = calc_fisher_diag(train_loader, model, criterion, optimizer)

    fisher_checkpoint = {
        'state_dict': model.state_dict(),
        'FI': fisher_diag,
        "config": config._asdict()
        }
    torch.save(fisher_checkpoint, get_filename_from_config(config, fisher=True))


def calc_fisher_diag(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()

    fisher_diag = {name: p.clone().zero_() for name, p in model.named_parameters()}
    number_of_samples = len(train_loader)

    # switch to train mode
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        # measure data loading time
        if config.gpu is not None:
            input_ = input_.cuda(config.gpu, non_blocking=True)
        target = target.cuda(config.gpu, non_blocking=True)

        # compute output
        output = model(input_)
        loss = criterion(output, target)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # Tracking the Expectation of the sum of parameters
        for name, parameter in model.named_parameters():
            fisher_diag[name] += (parameter.pow(2) / number_of_samples)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   batch_time=batch_time
            ))
    return fisher_diag




