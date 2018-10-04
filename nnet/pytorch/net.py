import numpy as np

import torch

from .models import ConvModel
from .base import BaseNet

__all__ = [
    'Net'
]

class Net(BaseNet):
    """Pytorch neural network."""

    def __init__(self, game, **kwargs):
        """
        Initializes the pytorch network. The input paramters define the network
        by making use of lists, such as strides=[3, 4, 5, 6], which would auto-
        matically create four sequential conv networks with stride 3, 4, 5, etc.

        Params
        ------

        game: Game
            The game class. Should extend from games.wrappers.Game.

        filters: list, default = [128, 256, 512, 512, 512, 512, 512, 512]
            The convolutional filters to apply. Applies them sequentially

        kernel: list, default = [5, 5, 5, 3, 3, 3, 3, 3]
            Kernel size in each conv layer

        strides: list, default = [1, 1, 1, 1, 1, 1, 1]
            The strides for each conv layer

        pad: lisdt, default = [2, 2, 2, 1, 1, 0, 0, 0]
            The padding for each convolution.

        fc: list, default = [1024]
            The number of neurons in the fully connected layers

        dropout: list, default = [0.2]
            The dropout to apply to each fc layer
        """
        super(Net, self).__init__(game)
        self.filters = kwargs.get('filters', [128, 256, 512, 512, 512, 512, 512, 512])
        self.kernel = kwargs.get('kernel', [5, 5, 5, 3, 3, 3, 3, 3])
        self.strides = kwargs.get('strides', [1, 1, 1, 1, 1, 1, 1, 1])
        self.pad = kwargs.get('pad', [2, 2, 2, 1, 1, 0, 0, 0])
        self.fc = kwargs.get('fc', [1024])
        self.dropout = kwargs.get('dropout', [0.2])
        

        if len(self.filters) == len(self.kernel) == len(self.strides) == len(self.pad) is False:
            raise ValueError('Filter, kernels and strides do not have the same shape')

        if len(self.fc) == len(self.dropout) is False:
            raise ValueError('Fc and dropout do not have the same shape')

    def build(self):
        self.model = ConvModel(self.game, self.filters, self.kernel, self.strides, self.pad, self.fc, self.dropout)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Parameters: %d' %params)
        if self.cuda:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.2)
        milestones = [self.step_size*(i+1) for i in range(3)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, 0.1)