import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Utils.fisherUtils import calc_fisher_utils, LossWithFisher
from config import config
from Utils import get_filename_from_config
from Utils.dataset import ClassDataset
from train import train, validate


def fisher():
    print('starting fisher calc')
    model, fisher_diag, star_params = calc_fisher_utils(filename=config.fisher_base_model)

    criterion = nn.CrossEntropyLoss().cuda(config.gpu)
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),
                             (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(config.data, train=True, download=True,
                                        transform=data_transforms)
    transfer_classes = list(set(range(10)) - set(config.classes))

    train_dataset = ClassDataset(transfer_classes, ds=full_train_dataset,
                                 transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    full_val_dataset = datasets.MNIST(config.data, train=False,
                                      download=True,
                                      transform=data_transforms)

    val_dataset = ClassDataset(transfer_classes, train=False,
                               ds=full_val_dataset,
                               transform=data_transforms)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True)

    old_dataset = datasets.MNIST(config.data, train=False,
                                      download=True,
                                      transform=data_transforms)

    old_train_dataset = ClassDataset(config.classes, train=False,
                               ds=old_dataset,
                               transform=data_transforms)

    old_train_loader = torch.utils.data.DataLoader(old_train_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True)

    criterion = LossWithFisher(criterion, model, fisher_diag, star_params, config.lam)

    scheduled_actions = None
    for epoch in range(config.start_epoch, config.epochs):
        criterion.set_train()
        # train for one epoch
        train((train_loader, old_train_loader), model, criterion, optimizer, epoch, scheduled_actions)
        criterion.set_validation()
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)


        torch.save({
            'epoch': epoch + 1,
            "config": config._asdict(),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(config.models, "checkpoint", get_filename_from_config(config)))

    torch.save({
        'epoch': epoch + 1,
        "config": config._asdict(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(config.models, "completed", get_filename_from_config(config)))
