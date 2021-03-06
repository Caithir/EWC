
import torch
# import torchvision.models as models
from arch import arches


def get_model_from_config(config):
    return arches[config.arch](num_classes=config.dataset_classes)


def get_filename_from_config(config, fisher=None, standard=None):
    """ will handle fisher file name here"""
    filename = "_".join([k+"-"+str(v) for k, v in config.relevant_params.items()])

    if standard:
        filename = f"{config.relevant_params['arc']}_cl-{config.relevant_params['cl']}_{config.relevant_params['dataset']}"
    if fisher:
        filename += "_FI"
    return f"{filename}.pth"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    pass


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

