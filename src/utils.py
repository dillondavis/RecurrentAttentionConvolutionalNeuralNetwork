"""Utils for PyTorch models and training"""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer
