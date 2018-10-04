import pickle
import numpy as np
import datetime
from .utils import get_data
import torch

class NetCoach(object):
    """
    Trains a player using MCTS.
    """

    def __init__(self, mcts, net, train_kw = None, max_moves=2000,
                 buffer=20, episodes=20, iterations = 100, train_time=0, 
                 prop_thresh=30, verbose=1, resume=False):
        self.mcts = mcts
        self.net = net
        self.nnet = True
        self.train_kw = train_kw
        self.max_moves = max_moves
        self.data_history = []
        self.buffer = buffer
        self.episodes = episodes
        self.iterations = iterations
        self.train_time = train_time
        self.calculation_time = datetime.timedelta(seconds=train_time)
        self.prop_thresh = prop_thresh
        self.verbose = verbose
        if resume:
            self.net.load_model()
            with open('./temp/data_hist.pkl', 'rb') as fp:
                self.data_history = pickle.load(fp)

    def _pprint(self, msg):
        if self.verbose:
            print(msg)

    def episode(self):
        """Executes an episode, defined as playing out a full game."""
        data, moves = get_data(self.mcts, nnet=self.nnet, max_moves=self.max_moves, return_moves=True, prop_thresh=self.prop_thresh)
        return data, moves

    def iteration(self):
        """Executes one iteration of the training loop."""
        train_data = []
        
        ep_times = []
        self._pprint('SELF PLAY FOR %d GAMES' %(self.episodes))
        for episode in range(self.episodes):
            ep_start = datetime.datetime.utcnow()
            ep_data, moves = self.episode()
            train_data.extend(ep_data)
            
            time_delta = datetime.datetime.utcnow() - ep_start
            ep_times.append(time_delta.total_seconds())
            ave_time = np.mean(ep_times)
            eta = ave_time * (self.episodes - episode - 1)

            self._pprint(
                'Game %d/%d finished, took %.2f seconds (%d moves). ETA: %.2f seconds' 
                %(episode+1, self.episodes, ep_times[-1], moves, eta)
            )
        self._pprint('')
        
        self.data_history.append(train_data)

        if len(self.data_history) > self.buffer:
            self._pprint("Memory capacity exceeded buffer, deleting oldest training data.\n")
            self.data_history.pop(0)

        train_data = []
        for data in self.data_history:
            train_data.extend(data)

        self._pprint("TRAINING ON %d BOARDS FROM MEMORY" %(len(train_data)))
        if self.train_kw is None:
            self.net.train(train_data)
        else:
            self.net.train(train_data, **self.train_kw)
        # reset the mcts after training
        self.mcts.reset()
        # save the model
        self.net.save_model()
        # save training data
        with open('./temp/data_hist.pkl', 'wb') as fp:
            pickle.dump(self.data_history, fp)
        self._pprint('')

    def train(self):
        """Trains the network over all iterations, or training time if not zero"""
        if self.train_time <= 0:
            # train on episodes
            begin = datetime.datetime.utcnow()
            
            time = []
            for i in range(self.iterations):
                i_begin = datetime.datetime.utcnow()
                self.iteration()
                i_end = datetime.datetime.utcnow()

                delta = i_end - i_begin
                delta = delta.total_seconds()
                time.append(delta)
                eta = np.mean(time)*(self.iterations - i)

                self._pprint("ITERATION %d done in %.2f seconds, ETA %.2f seconds\n" %(i+1, delta, eta))

            end = datetime.datetime.utcnow()
            delta = end - begin
            delta = delta.total_seconds()
            self._pprint("Done training, took %.2f seconds" %(delta))

        else:
            time = []
            begin = datetime.datetime.utcnow()
            i = 0
            while datetime.datetime.utcnow() - begin < self.calculation_time:
                i_begin = datetime.datetime.utcnow()
                self.iteration()
                i_end = datetime.datetime.utcnow()

                delta = i_end - i_begin
                delta = delta.total_seconds()
                time.append(delta)
                ave_time = np.mean(time)
                total = np.round(self.calculation_time.total_seconds()/ave_time)
                i += 1

                self._pprint("Iteration %d done in %.2f seconds, ESTIMATED ITERATIONS: %d\n" %(i, delta, total))

            end = datetime.datetime.utcnow()
            delta = end - begin
            delta = delta.total_seconds()
            self._pprint("Done training, took %.2f seconds" %(delta))
