# File containing the various models that can be used

from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, Activation, Add
from keras.activations import relu
from keras.engine.topology import Layer


class AlphaZero():
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
        action_size = game.action_size()

        # initial conv block
        inp = Input(shape=board_size)
        x = Conv2D(block_filters, block_kernel, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # residual blocks

        for i in range(blocks):
            # TODO: compress the residual block into a single layer
            res = Conv2D(block_filters, block_kernel, padding='same')(x)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Conv2D(block_filters, block_kernel, padding='same')(res)
            res = BatchNormalization()(res)
            res = Add()([res, x])
            x = Activation('relu')(res)

        # policy head
        p = Conv2D(policy_filters, kernel_size=1)(x)
        p = BatchNormalization()(p)
        p = Activation('relu')(p)
        p = Flatten()(p)
        p = Dense(action_size, activation='softmax', name='policy')(p)

        # value head
        v = Conv2D(value_filters, kernel_size=1)(x)
        v = BatchNormalization()(v)
        v = Activation('relu')(v)
        v = Flatten()(v)
        v = Dense(value_hidden, activation='relu')(v)
        v = Dense(1, activation='tanh', name='value')(v)

        self.model = Model(inputs = inp, outputs=[p, v])

from keras.backend import conv2d, batch_normalization, relu

class ResBlock(Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ResBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = Activation('relu')
        super(ResBlock, self).build(input_shape)

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape