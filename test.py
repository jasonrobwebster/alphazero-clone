import numpy as np

from montecarlo import MCTS
from games import Chess, TicTacToe
from players import NetPlayer, RandomPlayer
from coach import NetCoach
from nnet.pytorch import AlphaZeroNet

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from time import sleep
import datetime

game = Chess()
print(game.board())
print('')

player = RandomPlayer(game)
mcts = MCTS(game, player, episodes=400, verbose=1)
max_moves = 150

#coach = NetCoach(game, player, episodes=50, buffer=50, max_moves=max_moves, train_time=72*60*60)
#coach = NetCoach(game, player, max_moves=10, episodes=1, iterations=2, buffer=2, train_time=300)
#coach.train()


def make_mcts(output, game, model, optimizer, done_event):
    net = AlphaZeroNet(game, model=model, optimizer=optimizer, save_path='./models/pytorch/chess.model')
    player = player = NetPlayer(game, net)
    mcts = MCTS(game, player, episodes=400, verbose=1)
    mcts.train()
    output.put(mcts.get_policy())
    done_event.set()
    del mcts

if __name__ == '__main__':
    # Setup a list of processes that we want to run
    b = datetime.datetime.utcnow()
    mp.set_start_method('forkserver')
    output = mp.Queue()
    game = Chess()
    player = RandomPlayer(game)
    net = AlphaZeroNet(game, save_path='./models/pytorch/chess.model')
    net.build()
    model = net.model
    optim = net.optimizer
    model.share_memory()
    player = NetPlayer(game, net)
    processes = []
    events=[]

    # Run processes
    n = 3
    for rank in range(n):
        done_event = mp.Event()
        p = mp.Process(target=make_mcts, args=(output,game,model,optim,done_event))
        p.start()
        processes.append(p)
        events.append(done_event)
    for event in events:
        event.wait()
    print('Done')
    # Exit the completed processes
    #for p in processes:
    #    p.join()

    # Get process results from the output queue
    print('Done')
    results = [output.get() for p in processes]

    e = datetime.datetime.utcnow()
    print(results, (e-b)/n)


