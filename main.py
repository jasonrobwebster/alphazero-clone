import numpy as np

from montecarlo import MCTS, MCTSNet
from games import Chess, TicTacToe
from players import NetPlayer
from coach import NetCoach
from nnet.pytorch import AlphaZeroNet

from time import sleep

#game = Chess()
game = TicTacToe(5)
print(game.board())
print('')
net = AlphaZeroNet(game, blocks=10, epochs=10, save_path='./models/pytorch/chess.model')
net.build()
player = NetPlayer(game, net)
mcts = MCTSNet(game, player=player, episodes=400)
#mcts = MCTS(game, episodes=1000)
max_moves = 100

coach = NetCoach(mcts, net, episodes=200, buffer=50, max_moves=max_moves, train_time=17*60*60, prop_thresh=15, train_kw={'epochs': 10})
#coach = NetCoach(game, player, max_moves=10, episodes=1, iterations=2, buffer=2, train_time=300)
coach.train()

move = 0

def play(t):
    global move
    mcts = MCTS(game, player, train_time=t)
    mcts.train()
    prop = int(move<10)
    act = mcts.get_policy(prop=prop)
    act = np.random.choice(range(game.action_size()), p=act)
    game.move(act)
    move+=1
    print(move)
    print(game.board())
    print('')
