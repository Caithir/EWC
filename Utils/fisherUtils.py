import time
import os
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from Utils.dataset import ClassDataset
from .utils import get_model_from_config, get_filename_from_config, AverageMeter
from config import config, restore_config_from_dict

def calc_fisher_utils(model=None, filename=None):

    if not model:
        if not filename:
            raise ValueError("Must have either filename or model for fisher calc")
        checkpoint = torch.load(os.path.join(config.models,filename))
        config_from_file = restore_config_from_dict(checkpoint["config"])
        # Already calculated all the static fisher data for the given config
        if os.path.isfile(get_filename_from_config(config_from_file, fisher=True)):
            print("Fisher stats found for the model")
            model = get_model_from_config(config_from_file)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(config.gpu)
            fisher_diag = checkpoint['FI']
            star_params = {name: p.clone() for name, p in model.named_parameters()}
            return model, fisher_diag, star_params

        model = get_model_from_config(config_from_file)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # not making any steps, using to clear gradients
    optimizer = torch.optim.SGD(model.parameters(), 0)

    data_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))
                                   ])

    full_train_dataset = datasets.MNIST(config.data, train=True,
                                        download=True,
                                        transform=data_transforms)

    train_dataset = ClassDataset(config.classes, ds=full_train_dataset,
                                 transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=True)

    fisher_diag = calc_fisher_diag(train_loader, model, criterion, optimizer)
    print('diag calc done')
    fisher_checkpoint = {
        'state_dict': model.state_dict(),
        'FI': fisher_diag,
        "config": config._asdict()
        }
    print(f"model saved to :{get_filename_from_config(config, fisher=True)}")
    torch.save(fisher_checkpoint, get_filename_from_config(config, fisher=True))
    star_params = {name: p.clone().zero_() for name, p in model.named_parameters()}
    return model, fisher_diag, star_params


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
    with tqdm(enumerate(train_loader), total=len(train_loader)) as pb:
        for i, (input_, target) in pb:
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

            # if i % config.print_freq == 0:
            #     print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
            #            batch_time=batch_time
            #     ))
    return fisher_diag



class LossWithFisher(object):

    def __init__(self, criterion, model, fisher_diag, star_params):
        self.losses = [criterion, FisherPenalty(model, fisher_diag, star_params)]

    def __call__(self, output, target):
        loss_values = [loss(output, target) for loss in self.losses]
        return reduce(lambda x, y: x+y, loss_values)

class FisherPenalty(object):

    def __init__(self, model, fisher_diag, star_params):
        self.model = model
        self.fisher_diag = fisher_diag
        self.star_params = star_params

    def __call__(self, output, target):
        loss = torch.zeros(1)
        for n, p in self.model.named_parameters():
            _loss = self.fisher_diag[n] * (p - self.star_params[n]) ** 2
            loss += _loss.sum()
        return loss


