import time
import os
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable as V
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from Utils.dataset import ClassDataset
from .utils import get_model_from_config, get_filename_from_config, AverageMeter
from config import config, restore_config_from_dict
from .logger import logger

def calc_fisher_utils(model=None, filename=None):

    if not model:
        if not filename:
            raise ValueError("Must have either filename or model for fisher calc")

        # Already calculated all the static fisher data for the given config
        potential_path = os.path.join(config.models, 'completed', config.fisher_base_model[:-4] + '_FI.pth')
        if os.path.isfile(potential_path):
            print("Fisher stats found for the model")
            checkpoint = torch.load(potential_path)
            model = get_model_from_config(restore_config_from_dict(checkpoint['config']))
            model.load_state_dict(checkpoint['state_dict'])
            model.to(config.gpu)
            fisher_diag = checkpoint['FI']
            star_params = {name: p.clone() for name, p in model.named_parameters()}
            return model, fisher_diag, star_params


        checkpoint = torch.load(os.path.join(config.models, 'completed', filename))

        config_from_file = restore_config_from_dict(checkpoint["config"])
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
    checkpoint["FI"] = fisher_diag
    print(f"model saved to: {filename}")

    torch.save(checkpoint, os.path.join(config.models, "completed", filename[:-4]+"_FI.pth"))
    star_params = {name: p.clone().zero_() for name, p in model.named_parameters()}
    return model, fisher_diag, star_params


def calc_fisher_diag(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()

    fisher_diag = {name: p.clone().zero_().detach() for name, p in model.named_parameters()}
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
            for name, param in model.named_parameters():
                fisher_diag[name] += param.pow(2) / number_of_samples

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % config.print_freq == 0:
            #     print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
            #            batch_time=batch_time
            #     ))
    for name, param in fisher_diag.items():
        for p in param.view(-1):
            logger.log_fisher_diag(p.item())
    return fisher_diag


class LossWithFisher(object):

    def __init__(self, criterion, model, fisher_diag, star_params, lam, valida=False):
        self.losses = [criterion, FisherPenalty(model, fisher_diag, star_params, lam=lam)]
        self.valida = valida

    def set_validation(self):
        self.valida = True

    def set_train(self):
        self.valida = False

    def __call__(self, output, target):
        loss_values = [loss(output, target) for loss in self.losses]
        tag = 'val' if self.valida else ''
        logger.log_loss_componets(loss_values[0], "CE"+tag)
        logger.log_loss_componets(loss_values[1], "FI"+tag)

        return sum(loss_values)


class FisherPenalty(nn.Module):

    def __init__(self, model, fisher_diag, star_params, lam):
        super(FisherPenalty, self).__init__()
        self.model = model
        self.fisher_diag = fisher_diag
        self.star_params = star_params
        self.lam = lam

    def forward(self, output, target):
        loss = []
        for n, p in self.model.named_parameters():
            loss.append((self.lam * self.fisher_diag[n] * (p - self.star_params[n]) ** 2).sum())
        return sum(loss)


