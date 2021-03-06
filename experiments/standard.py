import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from configs.config import config
from Utils import get_model_from_config, get_filename_from_config
from Utils.dataset import ClassDataset
from train import train, validate


def standard():
    model = get_model_from_config(config)
    model.to(config.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(config.gpu)
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    sched = ReduceLROnPlateau(optimizer, factor=.2)

    # Standard set of data transformations for
    # use in all dataloaders
    data_transforms = transforms.Compose(config.experiment_transforms)

    kwargs = {
        'root': config.data,
        'train': True,
        'download': True,
        'transform': data_transforms
    }
    if config.dataset == datasets.EMNIST:
        kwargs['split'] = config.EMNIST_split

    full_train_dataset = config.dataset(**kwargs)

    train_dataset = ClassDataset(config.classes, ds=full_train_dataset,
                                 transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    kwargs['train'] = False
    full_val_dataset = config.dataset(**kwargs)

    val_dataset = ClassDataset(config.classes, train=False,
                               ds=full_val_dataset,
                               transform=data_transforms)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True)

    for epoch in range(config.start_epoch, config.epochs):

        train((train_loader,), model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        sched.step(prec1, epoch)
        torch.save({
            'epoch': epoch + 1,
            "config": config._asdict(),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(config.models, 'checkpoint', get_filename_from_config(config)))

    print(f"saving completed to: {os.path.join(config.models, 'completed', get_filename_from_config(config, standard=True))}")
    torch.save({
        "config": config._asdict(),
        'state_dict': model.state_dict(),
    }, os.path.join(config.models, 'completed', get_filename_from_config(config, standard=True)))
