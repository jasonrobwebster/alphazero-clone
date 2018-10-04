import numpy as np
from games import TicTacToe
from players import RandomPlayer, MCTSPlayer
from montecarlo import MCTS

game = TicTacToe(3)
print(game.board())

player_random = RandomPlayer(game)
mcts_play = MCTS(game=game, player=player_random, episodes=800)
player = MCTSPlayer(game=game, mcts=mcts_play)

mcts = MCTS(game=game, player=player_random, episodes=800)


for i in range(1000):
    while game.winner() == 0:
        mcts.train()
        p = mcts.get_policy(prop=1)
        a = np.random.choice(mcts.action_space, p=p)

        print(p, a, mcts.get_Qsa(game.state(), mcts.action_space), sum(mcts.get_Nsa(game.state(), mcts.action_space)))

        game.move(a)

        print(game.board())
        mcts.update()

    game.set_state(game.start())
    print(i)
    print(game.board())
    print()
    mcts.update()