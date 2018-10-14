import numpy as np

from montecarlo import MCTS, MCTSNet
from games import Chess, TicTacToe, Othello
from players import NetPlayer
from nnet.keras import AlphaZeroNet

#game = Chess()
#game = TicTacToe(3)
game = Othello(6)
print(game.board())
print()

net = AlphaZeroNet(game, blocks=10, epochs=10, save_path='./models/keras/othello2.model')
net.load_model()

player = NetPlayer(game, net)
move = 0

def play(t):
    global move

    """legal_moves =game.engine.legal_moves()
    print(legal_moves)
    human = input()
    x, y = human.split(' ')
    x, y = int(x), int(y)
    while (x, y) not in legal_moves:
        human = input()
        x, y = human.split(' ')
        x, y = int(x), int(y)
    game.move((x, y))
    move +=1
    print(game.board())
    print('')"""

    mcts = MCTSNet(game, player, episodes=100, exploration=0.5)
    mcts.train()
    prop = int(move<0)
    act = mcts.get_policy(prop=prop)
    act = np.random.choice(range(game.action_size()), p=act)
    s = game.state()
    game.move(act)
    print(mcts.get_Qsa(s, act), mcts.get_Nsa(s, act))
    move += 1
    print(game.board())
    print('')

    if game.winner() != 0:
        return

    mcts_random = MCTS(game, episodes=100)
    mcts_random.train()
    prop = int(move<0)
    act = mcts_random.get_policy(prop=prop)
    act = np.random.choice(range(game.action_size()), p=act)
    s = game.state()
    game.move(act)
    print(mcts_random.get_Qsa(s, act), mcts_random.get_Nsa(s, act))
    move += 1
    print(game.board())
    print('')


    if game.winner() != 0:
        return
    

while game.winner() == 0:
    play(1)

print(game.result())