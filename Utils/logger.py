import math
import os
from collections import Counter
import torch
from tensorboardX import SummaryWriter
from config import config, log_items
from Utils import get_filename_from_config


class Logger(object):

    def __init__(self):
        self.writer = SummaryWriter(os.path.join(config.logs, config.experiment_name,
                                                 get_filename_from_config(config)[:-4]))
        self.log_counters = Counter()
        self.tag_prefix = os.path.join("")

    def log(self, tag, value):
        if type not in log_items:
            raise ValueError(f"Cannot log {tag}")
        tag = os.path.join(self.tag_prefix, tag)
        self.writer.add_scalar(tag, value, self.log_counters[tag])

    def train_batch_log(self, model, accuracy, loss):
        grad_max, grad_min = -math.inf, math.inf

        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p_max = torch.max(p.grad.data)
            p_min = torch.min(p.grad.data)
            grad_max = max(p_max.item(), grad_max)
            grad_min = min(p_min.item(), grad_min)
        grad_min_tag = str(os.path.join(self.tag_prefix, 'grad',  'min'))
        grad_max_tag = str(os.path.join(self.tag_prefix, 'grad',  'max'))
        self.writer.add_scalar(grad_min_tag, grad_min, self.log_counters[grad_min_tag])
        self.writer.add_scalar(grad_max_tag, grad_max, self.log_counters[grad_max_tag])
        self.log_counters[grad_min_tag] += 1
        self.log_counters[grad_max_tag] += 1

        train_main_tag = str(os.path.join(self.tag_prefix, 'train'))
        train_acc_tag = str(os.path.join(train_main_tag, 'accuracy'))
        train_loss_tag = str(os.path.join(train_main_tag, 'loss'))
        self.writer.add_scalar(train_loss_tag, loss, self.log_counters[train_loss_tag])
        self.writer.add_scalar(train_acc_tag, accuracy, self.log_counters[train_acc_tag])

        self.log_counters[train_acc_tag] += 1
        self.log_counters[train_loss_tag] += 1

    def val_batch_log(self, accuracy, loss, fisher=False):
        val_main_tag = str(os.path.join(self.tag_prefix, 'val'))
        suffix_tag = ""
        # For overlaying the validation on previous task with current task
        if fisher:
            val_main_tag = str(os.path.join(self.tag_prefix, 'train'))
            suffix_tag = "fisher"
        val_acc_tag = str(os.path.join(val_main_tag, 'accuracy', suffix_tag))
        val_loss_tag = str(os.path.join(val_main_tag, 'loss', suffix_tag))
        self.writer.add_scalar(val_acc_tag, accuracy, self.log_counters[val_acc_tag])
        self.writer.add_scalar(val_loss_tag, loss, self.log_counters[val_loss_tag])

        self.log_counters[val_acc_tag] += 1
        self.log_counters[val_loss_tag] += 1

    def log_fisher_diag(self, diag_val):
        self.writer.add_scalar("Fisher Diag", diag_val, self.log_counters["fisher_diag"])
        self.log_counters['fisher_diag'] += 1

    def log_loss_componets(self, val, name):
        self.writer.add_scalar("loss/"+name, val, self.log_counters[name])
        self.log_counters[name] += 1


logger = Logger()