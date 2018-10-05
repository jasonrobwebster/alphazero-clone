import os
import sys
import numpy as np

from ..wrappers import NNetWrap

__all__ = [
    'BaseNet'
]

class BaseNet(NNetWrap):
    """Keras base neural network. Prevents writing the same code multiple times by providing
    an already working training, prediction, and save/load state methods."""

    def __init__(self, **kwargs):
        self.loss = ['categorical_crossentropy', 'mean_squared_error']
        self.model = kwargs.get('model', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.path = kwargs.get('save_path', './models/keras/keras.model')
        self.step_size = kwargs.get('step_size', 15)
        self.callbacks = None

    def train(self, data, epochs=5, batch_size=32, shuffle=True):
        """
        Trains the model. Data should be a list of the form [(state, target_pi, target_v), ...].
        The variable 'state' should already be in the form acceptable to the net, so use
        game.represent_nn().

        Params
        ------

        data: list 
            List of data containing training exampls of the form [(state, target_pi, target_v), ...].
        
        epochs: int, default = 5
            The number of epochs to train the model over.

        batch_size: int, default = 32
            The batch size to train on.

        shuffle: bool, default=True
            Whether the shuffle the data at the start of an iteration.

        Returns
        -------

        hist: History object
            A history object containing the history of the loss, pi_loss and v_loss, as attributes.
        """

        if self.model is None:
            self.build()
        if self.optimizer is None:
            raise TypeError('Optimizer has not been set!')

        # get the data
        states, pis, vs = list(zip(*data))
        states = np.asarray(states)
        pis = np.asarray(pis)
        vs = np.asarray(vs)

        history = self.model.fit(
            x=states, y=[pis,vs],
            epochs=epochs,
            batch_size=batch_size,
            verbose=True,
            shuffle=shuffle,
            callbacks=self.callbacks
        )

        return history

    def predict(self, state):
        if self.model is None:
            self.build()

        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)
        
        policy, v = self.model.predict_on_batch(state)
        return policy[0], v[0][0]

    def save_model(self):
        if self.model is None:
            self.build()

        self.model.save_weights(self.path)

    def load_model(self):
        if self.model is None:
            self.build()

        self.model.load_weights(self.path)

        