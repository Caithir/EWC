import torch

from torch.nn.utils import clip_grad_value_
from .logger import logger


def clip_and_track_grad(model, config):
    epsilon = pow(1/10, 3)
    clipped_params = {name: p.grad.data.clone().detach()
                      for name, p in model.named_parameters()}
    # clip_grad_value_(model.parameters(), config.grad_clip)
    clip_batch_norm_grad_value_(model.named_parameters(), config.grad_clip)
    for name, param in model.named_parameters():
        clipped_params[name] -= param.grad.data
        if torch.max(clipped_params[name]) < epsilon or torch.min(clipped_params[name]) > -epsilon:
            del clipped_params[name]
        else:
            clipped_params[name] = clipped_params[name].view(-1).cpu().numpy()

    logger.log_grad_graph(clipped_params)


def clip_batch_norm_grad_value_(named_params, clip_value):

    clip_value = float(clip_value)
    for name, p in filter(lambda p: p[1].grad is not None or "bn" in p[0], named_params):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)
