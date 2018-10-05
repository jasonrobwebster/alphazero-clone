import numpy as np

from keras.layers import (
    Input, Conv2D, Reshape, 
    Flatten, BatchNormalization, 
    Dropout, Dense)
from keras.models import Model

from ..wrappers import NNetWrap

class Net(NNetWrap):
    """Keras neural network."""

    def __init__(self, game, **kwargs):
        """
        Initializes the keras network. The input paramters define the network
        by making use of lists, such as strides=[3, 4, 5, 6], which would auto-
        matically create four sequential conv networks with stride 3, 4, 5, etc.

        Params
        ------

        game: Game
            The game class. Should extend from games.wrappers.Game.

        filters: list, default = [256, 256, 256, 256]
            The convolutional filters to apply. Applies them sequentially

        kernel: list, default = [3, 3, 3, 3]
            Kernel size in each conv layer

        strides: list, default = [1, 1, 1, 1]
            The strides for each conv layer

        fc: list, default = [1024, 512]
            The number of neurons in the fully connected layers

        dropout: list, default = [0.2, 0.2]
            The dropout to apply to each fc layer
        """

        self.game = game
        self.filters = kwargs.get('filters', [256, 256, 256, 256])
        self.kernel = kwargs.get('kernel', [3, 3, 2, 2])
        self.strides = kwargs.get('strides', [1, 1, 1, 1])
        self.fc = kwargs.get('fc', [1024, 512])
        self.dropout = kwargs.get('dropout', [0.2, 0.2])
        self.model = None

        if len(self.filters) == len(self.kernel) == len(self.strides) is False:
            raise ValueError('Filter, kernels and strides do not have the same shape')

        if len(self.fc) == len(self.dropout) is False:
            raise ValueError('Fc and dropout do not have the same shape')

    def build(self):
        """Builds the network."""
        board_size = self.game.get_board_size(nnet=True)
        action_size = self.game.action_size()

        self.input_board = Input(shape=board_size, name='input')
        
        # conv network
        bn = self.input_board
        for i in range(len(self.filters)):
            conv = Conv2D(
                self.filters[i], 
                kernel_size=self.kernel[i], 
                strides=self.strides[i], 
                activation='relu', 
                name='conv_%d' %(i+1))(bn)
            bn = BatchNormalization(axis=3)(conv)

        # fc network
        flat = Flatten()(bn)
        drop = flat
        for i in range(len(self.fc)):
            fc = Dense(self.fc[i], activation='relu', name='dense_%d' %(i+1))(drop)
            bn = BatchNormalization()(fc)
            drop = Dropout(self.dropout[i], name='dropout_%d' %(i+1))(bn)

        # outputs
        self.pi = Dense(action_size, name='pi', activation='softmax')(drop)
        self.v = Dense(1, name='value', activation='tanh')(drop)
        
        # model
        self.model = Model(self.input_board, [self.pi, self.v], name='keras_net')
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')

    def train(self, data, **kwargs):
        """
        Train the model. The kwargs are passed to the model.fit function, and
        should contain the epochs, batch_size, etc, parameters.
        Data should be a list of the form [(state, target_pi, target_v), ...].
        The variable 'state' should already be in the form acceptable to the net.

        Returns
        -------

        history: Keras History object
            The keras history object, returned by model.fit.
        """
        if self.model is None:
            self.build()

        # get keras kwargs
        epochs = kwargs.get('epochs', 2)
        batch_size = kwargs.get('batch_size', 32)
        verbose = kwargs.get('verbose', 1)
        callbacks = kwargs.get('callbacks', None)
        validation_split = kwargs.get('validation_split', 0.0)
        validation_data = kwargs.get('validation_data', None)
        shuffle = kwargs.get('shuffle', True)
        class_weight = kwargs.get('class_weight', None)
        sample_weight = kwargs.get('sample_weight', None)
        initial_epoch = kwargs.get('initial_epoch', 0)
        steps_per_epoch = kwargs.get('steps_per_epoch', None)
        validation_steps = kwargs.get('validation_steps', None)

        # get the data
        states, pis, vs = list(zip(*data))
        states = np.asarray(states)
        pis = np.asarray(pis)
        vs = np.asarray(vs)

        history = self.model.fit(
            x=states, y=[pis,vs],
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )

        return history


    def predict(self, state):
        if self.model is None:
            self.build()

        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)

        policy, v = self.model.predict_on_batch(state)
        if len(state) == 1:
            return policy[0], v[0][0]
        else:
            return policy, v
        

