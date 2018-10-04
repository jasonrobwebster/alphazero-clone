import numpy as np

from .wrappers import Player

class RandomPlayer(Player):
    """Randomly chooses a move."""

    def __init__(self, game):
        self.game = game

    def action(self):
        moves = self.game.legal_moves()
        moves = np.argwhere(moves == 1)
        move = np.random.choice(moves.reshape(len(moves)))
        return move

    def evaluate(self):
        moves = self.game.legal_moves()
        moves = moves / float(sum(moves))
        return moves, 0

class MCTSPlayer(Player):
    "Chooses a move by training an mcts"

    def __init__(self, game, mcts):
        self.game = game
        self.mcts = mcts

    def action(self):
        self.mcts.update()
        policy, _ = self.evaluate()
        best_act = policy.argmax()
        best_acts = np.argwhere(policy == policy[best_act])
        if len(best_acts) > 1:
            best_act = np.random.choice(best_acts.reshape(len(best_acts)))
        return best_act

    
    def evaluate(self):
        self.mcts.update()
        state = self.game.state()
        act_space = self.mcts.action_space
        Qsa = self.mcts.get_Qsa(state, act_space)
        value = sum(Qsa)
        policy = self.mcts.get_policy(prop=0)
        return policy, value


class NetPlayer(Player):
    "Neural network player. Net should extend the NNetBase class"

    def __init__(self, game, net):
        self.game = game
        self.net = net

    def action(self):
        policy, _ = self.evaluate()
        best_act = policy.argmax()
        best_acts = np.argwhere(policy == policy[best_act])
        if len(best_acts) > 1:
            best_act = np.random.choice(best_acts.reshape(len(best_acts)))
        return best_act

    def evaluate(self):
        state = self.game.represent_nn()
        policy, value = self.net.predict(state)

        legal_moves = self.game.legal_moves()
        policy = policy * legal_moves
        sum_p = np.sum(policy)
        
        if sum_p != 0:
            policy = policy / sum_p
        else:
            # policy is masking all legal moves
            print("WARNING: player policy has masked all legal moves")
            print("Making all legal moves equally likely")
            policy = policy + legal_moves
            policy = policy / np.sum(policy)
        return policy, value