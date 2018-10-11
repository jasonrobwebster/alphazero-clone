import numpy as np

from .engine import TicTacToeEngine
from ..wrappers import Game

class TicTacToe(Game):
    """A tic tac toe game."""

    def __init__(self, n=3):
        """Initialises the game. Sets the board size to nxn."""
        self.n = n
        self.engine = TicTacToeEngine(n)
        self.start_pos = self.engine.board_state()
        move_id = {(i, j): i*n+j for i in range(n) for j in range(n)}
        id_move = {i*n+j: (i, j) for i in range(n) for j in range(n)}
        self.move_id = move_id
        self.id_move = id_move

    def board(self):
        return self.engine.display()

    def get_board_size(self, nnet=True):
        if nnet:
            return self.engine.size + (2,)
        else:
            return self.engine.size

    def start(self):
        return self.start_pos

    def current_player(self):
        return self.engine.player

    def state(self, nnet=False):
        if nnet:
            return self.represent_nn()
        return self.engine.board_state()

    def canonical_state(self):
        return self.engine.board_state()

    def set_state(self, state):
        self.engine.set_board(state)

    def move(self, action):
        if isinstance(action, tuple):
            self.engine.move(action)
        else:
            # convert to tuple
            x = self.id_move.get(action, None)
            if x is None:
                raise ValueError('Given action %d does not have an associated move.' %action)    
            self.engine.move(x)

    def action_size(self):
        return self.n * self.n

    def legal_moves(self):
        """Returns the legal moves of the current game as a masking array."""
        moves = self.engine.legal_moves()
        moves = list(map(lambda move: self.move_id[move], moves))

        out = np.zeros(self.action_size())
        out[moves] = 1
        return out

    def winner(self):
        result = self.engine.result()
        if result == '*':
            # game still going
            return 0
        if result == '1-0' or result == '0-1':
            # it's always the turn of the loser at this point
            # so return the win value (for the player above us)
            return -1
        if result == '1/2-1/2':
            return 0.00001

    def result(self):
        return self.engine.result()

    def represent_nn(self, state=None):
        if state is not None:
            board_state = self.state()
            self.set_state(state)

        board_nn = self.engine.board
        out = np.zeros((*board_nn.shape, 2))

        # args where we have 1
        args_1 = np.argwhere(board_nn==1)
        for arg in args_1:
            x, y = arg
            z = 0
            out[x, y, z] = 1

        # args where we have -1
        args_m1 = np.argwhere(board_nn==-1)
        for arg in args_m1:
            x, y = arg
            z = 1
            out[x, y, z] = 1

        #board_nn = np.expand_dims(board_nn, 2)

        if state is not None:
            # reset state
            self.set_state(board_state)
            
        return out

    def get_symmetries(self, nnet=False):
        syms = self.engine.symmetries()
        if nnet:
            syms = list(map(lambda x: self.represent_nn(x), syms))
        return syms


    
            