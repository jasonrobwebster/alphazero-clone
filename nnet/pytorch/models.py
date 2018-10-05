import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import conv_out_shape


class AlphaZero(nn.Module):
    """The alphazero model."""
    
    def __init__(self, game, 
                 block_filters=256, 
                 block_kernel=3, 
                 blocks=20, 
                 policy_filters=2, 
                 value_filters=1, 
                 value_hidden=256):
        super(AlphaZero, self).__init__()
        board_size = game.get_board_size(nnet=True)
        board_size = (board_size[2], board_size[0], board_size[1]) # make channels first for pytorch
        action_size = game.action_size()

        # padding to get the same size output only works for odd kernel sizes
        if block_kernel % 2 != 1:
            raise ValueError('block_kernel must be odd, got %d' %block_filters)
        pad = int(np.floor(block_kernel/2))
        
        # the starting conv block
        self.conv_block = nn.Sequential(
            nn.Conv2d(board_size[0], block_filters, kernel_size=block_kernel, stride=1, padding=pad),
            nn.BatchNorm1d(num_features=block_filters),
            nn.ReLU()
        )

        # the residual blocks
        self.blocks = blocks
        self.res_blocks = nn.ModuleList([ResBlock(block_filters, block_kernel) for i in range(blocks-1)])

        # policy head
        self.policy_conv = nn.Conv2d(block_filters, policy_filters, kernel_size=1)
        self.policy_conv_bn = nn.BatchNorm1d(policy_filters)
        # calculate policy output shape to flatten
        pol_shape = (policy_filters, board_size[1], board_size[2])
        self.policy_flat = int(np.prod(pol_shape))
        # policy layers
        self.policy_bn = nn.BatchNorm1d(num_features=policy_filters)
        self.policy = nn.Linear(self.policy_flat, action_size)

        # value head
        self.value_conv = nn.Conv2d(block_filters, value_filters, kernel_size=1)
        self.value_conv_bn = nn.BatchNorm1d(value_filters)
        # calculate value shape to flatten
        val_shape = (value_filters, board_size[1], board_size[2])
        self.value_flat = int(np.prod(val_shape))
        # value layers
        self.value_hidden = nn.Linear(self.value_flat, value_hidden)
        self.value = nn.Linear(value_hidden, 1)

    def forward(self, x):
        x = self.conv_block(x)
        for i in range(self.blocks):
            x = self.res_blocks[i](x)
        
        # policy head
        x_pi = self.policy_conv(x)
        x_pi = self.policy_conv_bn(x_pi)
        x_pi = x_pi.view(-1, self.policy_flat)
        x_pi = F.relu(x_pi)
        x_pi = self.policy(x_pi)
        x_pi = F.softmax(x_pi)

        # value head
        x_v = self.value_conv(x)
        x_v = self.value_conv_bn(x_v)
        x_v = F.relu(x_v)
        x_v = x_v.view(-1, self.value_flat)
        x_v = self.value_hidden(x_v)
        x_v = F.relu(x_v)
        x_v = self.value(x_v)
        x_v = F.tanh(x_v)
        return x_pi, x_v

# Older models I tried out

class ConvModel(nn.Module):
    def __init__(self, game, filters, kernels, strides, pad, fc, dropout):
        super(ConvModel, self).__init__()
        board_size = game.get_board_size(nnet=True)
        board_size = (board_size[2], board_size[0], board_size[1]) # make channels first
        action_size = game.action_size()

        if len(filters) > 0:
            self.conv = nn.ModuleList([
                nn.Conv2d(
                    in_channels=board_size[0] if i==0 else filters[i-1],
                    out_channels=filters[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=pad[i]
                ) 
                for i in range(len(filters))
            ])
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(
                    num_features=filter
                )
                for filter in filters
            ])

            conv_shape = conv_out_shape(board_size, filters[0], kernels[0], strides[0], pad[0])
            # need the final conv_shape
            for i in range(len(filters)-1):
                conv_shape = conv_out_shape(conv_shape, filters[i+1], kernels[i+1], strides[i+1], pad[i+1])
            self.flat_out = int(np.prod(conv_shape))
        else:
            self.conv = []
            self.bn = []
            self.flat_out = int(np.prod(board_size))
        if len(fc) > 0:
            self.fc = nn.ModuleList([
                nn.Linear(
                    in_features=self.flat_out if i==0 else fc[i-1],
                    out_features=fc[i]
                )
                for i in range(len(fc))
            ])

            self.bn2 = nn.ModuleList([
                nn.BatchNorm1d(
                    num_features=feature
                )
                for feature in fc
            ])

            self.drop = nn.ModuleList([
                nn.Dropout(p=p) for p in dropout
            ])
        
            self.pi = nn.Linear(fc[-1], action_size)
            self.v = nn.Linear(fc[-1], 1)
        else:
            self.fc = []
            self.bn2 = []
            self.drop = []
            self.pi = nn.Linear(self.flat_out, action_size)
            self.v = nn.Linear(self.flat_out, 1)

    def forward(self, x):
        # do conv part
        for i in range(len(self.conv)):
            x = F.relu(self.conv[i](x))
            x = self.bn[i](x)
        
        x = x.view(-1, self.flat_out) # flatten
        # do fc part
        for i in range(len(self.fc)):
            x = F.relu(self.fc[i](x))
            x = self.bn2[i](x)
            x = self.drop[i](x)

        # do pi and v
        x_pi = self.pi(x)
        x_v = self.v(x)

        x_pi = F.softmax(x_pi)
        x_v = F.tanh(x_v)
        return x_pi, x_v

class ResBlock(nn.Module):
    def __init__(self, filters=256, kernel=3):
        super(ResBlock, self).__init__()
        if kernel % 2 != 1:
            raise ValueError('kernel must be odd, got %d' %kernel)
        pad = int(np.floor(kernel/2))

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(num_features=filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + inp
        x = F.relu(x)
        return x