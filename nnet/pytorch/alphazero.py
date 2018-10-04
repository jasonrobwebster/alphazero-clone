import numpy as np

import torch

from .models import AlphaZero
from .base import BaseNet

__all__ = [
    'Net'
]

class AlphaZeroNet(BaseNet):
    """Pytorch neural network."""

    def __init__(self, game, **kwargs):
        """
        Initializes the pytorch alphazero network. The input paramters are linked to
        the paper, see [1].

        Params
        ------

        game: Game
            The game class. Should extend from games.wrappers.Game.

        block_filters: int, default = 256
            The number of filters to use in the residual blocks.

        block_kernel: int, default = 3
            The kernel size in the residual blocks.

        blocks: int, default = 19
            The number of residual blocks to use.

        policy_filters: int, default = 2
            The number of filters in the policy head's convolutional filter.

        value_filters: int, default = 1
            The number of filters in the value head's convolutional filter.

        value_hidden: int, default = 256
            The size of the hidden layer in the value head.

        save_path: str, default = ./models/pytorch/pytorch.model
            The path to save the game in.
        """
        super(AlphaZeroNet, self).__init__(**kwargs)
        self.game = game
        self.block_filters = kwargs.get('block_filters', 256)
        self.block_kernel = kwargs.get('block_kernel', 3)
        self.blocks = kwargs.get('blocks', 19)
        self.policy_filters = kwargs.get('policy_filters', 2)
        self.value_filters = kwargs.get('value_filters', 1)
        self.value_hidden = kwargs.get('value_hidden', 256)

    def build(self):
        self.model = AlphaZero(
            self.game, 
            self.block_filters, 
            self.block_kernel, 
            self.blocks,
            self.policy_filters,
            self.value_filters,
            self.value_hidden
        )
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Parameters: %d' %params)
        if self.cuda:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)
        milestones = [self.step_size*(i+1) for i in range(3)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, 0.1)