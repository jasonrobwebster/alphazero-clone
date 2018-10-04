import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    'Model',
    'History',
    'cross_entropy',
    'conv_out_shape'
]

class History():
    def __init__(self, loss, pi_loss, v_loss):
        self.loss = loss
        self.pi_loss = pi_loss
        self.v_loss = v_loss

def cross_entropy(p, pi):
    _, action_size = pi.size()
    pi = pi.view(-1, 1, action_size)
    p = torch.log(p)
    p = p.view(-1, action_size, 1)
    out = -torch.bmm(pi, p)
    out = torch.mean(out)
    return out

def conv_out_shape(in_shape, out_features, kernel, stride, pad=0, dilation=(1,1)):
    """Calculates the output of a convolutional operation."""
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in = in_shape[1]
    w_in = in_shape[2]
    h_out = np.floor((h_in + 2*pad - dilation[0]*(kernel[0]-1)-1)/stride[0] + 1)
    w_out = np.floor((w_in + 2*pad - dilation[1]*(kernel[1]-1)-1)/stride[1] + 1)
    out_shape = (out_features, h_out, w_out)
    return out_shape