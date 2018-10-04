import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import cross_entropy, History
from ..wrappers import NNetWrap

__all__ = [
    'BaseNet'
]

class BaseNet(NNetWrap):
    """Pytorch base neural network. Helps prevent writing the same code multiple times by providing
    an already working training, prediction, and save/load state methods."""

    def __init__(self, **kwargs):
        """Should be called as a super().__init__(**kwargs) for child class."""
        self.loss = [cross_entropy, nn.MSELoss()]
        self.model = kwargs.get('model', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.path = kwargs.get('save_path', './models/pytorch/pytorch.model')
        self.cuda = torch.cuda.is_available()
        self.step_size = kwargs.get('step_size', 15)
        self.epochs = 0

    def build(self):
        """Should set two attributes: model and optimizer."""
        self.model = None
        self.optimizer = None
        raise NotImplementedError

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
            raise TypeError('Optimizer has not been set! Check the build method of your net.')

        self.model.train()

        # get the data
        states, pis, vs = list(zip(*data))
        states = np.asarray(states).transpose((0, 3, 1, 2)) #transpose to chanels first
        pis = np.asarray(pis)
        vs = np.asarray(vs)
        if shuffle:
            np.random.shuffle(states)
            np.random.shuffle(pis)
            np.random.shuffle(vs)

        states = torch.from_numpy(states).float()
        pis = torch.from_numpy(pis).float()
        vs = torch.from_numpy(vs).float()

        loss = []
        pi_loss = []
        v_loss = []
        for epoch in range(epochs):
            self.epochs += 1

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            batch = 0
            epoch_loss = 0
            epoch_pi_loss = 0
            epoch_v_loss = 0
            n_batches = 0
            
            while batch < len(data):
                batch_s = Variable(states[batch:batch+batch_size]) 
                batch_p = Variable(pis[batch:batch+batch_size])
                batch_v = Variable(vs[batch:batch+batch_size])
                batch += batch_size
                batch_len = len(batch_s)

                if self.cuda:
                    batch_s = batch_s.cuda()
                    batch_p = batch_p.cuda()
                    batch_v = batch_v.cuda()

                self.optimizer.zero_grad()

                policy, v = self.model(batch_s)
                loss_pi = self.loss[0](policy, batch_p)
                loss_v = self.loss[1](v, batch_v)
                total_loss = loss_pi + loss_v
                total_loss.backward()
                self.optimizer.step()
                
                epoch_pi_loss = (n_batches * epoch_pi_loss + loss_pi.data[0])/(1 + n_batches) # ave loss
                epoch_v_loss = (n_batches * epoch_v_loss + loss_v.data[0])/(1 + n_batches) #ave loss
                epoch_loss = epoch_pi_loss + epoch_v_loss

                percent = np.round(min(batch/len(data), 1)*100)

                print(
                    'EPOCH %d -- %d%% complete -- loss: %.4f - pi_loss: %.4f - v_loss: %.4f' 
                    %(self.epochs, percent, epoch_loss, epoch_pi_loss, epoch_v_loss), end='\r'
                )
                sys.stdout.flush()
                n_batches += 1

            loss.append(epoch_loss)
            pi_loss.append(epoch_pi_loss)
            v_loss.append(epoch_v_loss)
            print(
                'EPOCH %d -- %d%% complete-- loss: %.4f - pi_loss: %.4f - v_loss: %.4f COMPLETE' 
                %(self.epochs, percent, epoch_loss, epoch_pi_loss, epoch_v_loss)
            )
            sys.stdout.flush()
        # training done
        hist = History(loss, pi_loss, v_loss)
        return hist

    def predict(self, state):
        if self.model is None:
            self.build()
        
        self.model.eval()

        if len(state.shape) == 3:
            state = np.transpose(state, (2, 0, 1)) #transpose to channels first
            state = np.expand_dims(state, 0)
        state = torch.from_numpy(state)
        state = state.float()
        state = Variable(state)
        if self.cuda:
            state = state.cuda()
        policy, v = self.model(state)
        policy = policy.data[0].cpu().numpy()
        v = v.data[0].cpu().numpy()[0]
        return policy, v

    def save_model(self):
        if self.model is None:
            self.build()
        state = {
            'model' : self.model,
            'optimizer' : self.optimizer
        }
        torch.save(state, self.path)

    def load_model(self):
        loaded = torch.load(self.path)
        self.model = loaded['model']
        self.optimizer = loaded['optimizer']