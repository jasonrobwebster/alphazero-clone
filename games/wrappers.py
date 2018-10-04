__all__ = [
    'Game'
]

class Game(object):
    """
    This is the base class for a two player game.
    All games should derive from this class.
    """

    #def __init__(self):
    #    pass

    def board(self):
        """The current board for the game"""
        raise NotImplementedError

    def get_board_size(self, nnet=False):
        """Returns a tuple giving the size of the board. Returns the size if the nnet input if nnet=True"""
        raise NotImplementedError

    def start(self):
        """The starting position of the game"""
        raise NotImplementedError

    def current_player(self):
        """The current player, either 1 or 2 (or more for multiplayer games)"""
        raise NotImplementedError

    def state(self):
        """The current state of the game"""
        raise NotImplementedError

    def canonical_state(self):
        """The current state of the game in canonical form. Should encapsulate the state and the current
        player, and any other variable that may cause an automated agent to yield a unique answer."""
        raise NotImplementedError

    def set_state(self, state):
        """Sets the state of the board."""
        raise NotImplementedError

    def move(self, action):
        """Makes a move for the game given an action. Updates internal state and player."""
        raise NotImplementedError

    def action_size(self):
        """Returns the possible action size of the game"""
        raise NotImplementedError

    def legal_moves(self):
        """Returns the allowed legal moves of the game as a masking array"""
        raise NotImplementedError
        
    def winner(self):
        """Returns 1 if player won, -1 if lost, 0 if still going or drawn"""
        raise NotImplementedError

    def represent_nn(self, state):
        """Should return the matrix that can be fed into a neural network. Has to be of the form
        array(W x H x C) where W and H are the board width and height, and C are the channels used
        in convolutional neural nets."""
        pass

    def get_symmetries(self, nnet=False):
        """Return the symmetric states associated with this board."""
        if nnet:
            return (self.represent_nn(),)
        return self.state()