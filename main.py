import numpy as np

from montecarlo import MCTS, MCTSNet
from games import Chess, TicTacToe, Othello
from players import NetPlayer
from coach import NetCoach
from nnet.keras import AlphaZeroNet

#game = Chess()
#game = TicTacToe(3)
game = Othello(6)
print(game.board())
print()
net = AlphaZeroNet(game, blocks=10, epochs=10, save_path='./models/keras/othello2.model')
net.build()
player = NetPlayer(game, net)
mcts = MCTSNet(game, player=player, episodes=100, exploration=0.2)
#mcts = MCTS(game, episodes=100)
max_moves = 100

coach = NetCoach(mcts, net, episodes=100, buffer=50, max_moves=max_moves, train_time=14*60*60, prop_thresh=12, train_kw={'epochs': 10})
#coach = NetCoach(game, player, max_moves=10, episodes=1, iterations=2, buffer=2, train_time=300)
coach.train()


