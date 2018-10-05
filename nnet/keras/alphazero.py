# keras implementation of the AlphaZero model

from .models import AlphaZero
from .base import BaseNet

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

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
        self.alphazero = AlphaZero(
            self.game, 
            self.block_filters, 
            self.block_kernel, 
            self.blocks,
            self.policy_filters,
            self.value_filters,
            self.value_hidden
        )
        self.model = self.alphazero.model
        

        params = self.model.count_params()
        print('Parameters: %d' %params)

        lr = 1e-2
        milestones = [self.step_size*(i+1) for i in range(3)]
        def scheduler(epoch):
            for i, milestone in enumerate(milestones):
                if epoch <= milestone:
                    return lr * 10**(-i)

        lr_sched = LearningRateScheduler(scheduler)
        self.callbacks = [lr_sched]

        self.optimizer = Adam(lr=1e-2)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)