class Player(object):
    """
    Base class for players. Should be given a state and return an action.
    If the player is a neural net, __init__ should have a property 'is_nnet' set to True.
    """

    def __init__(self, game):
        pass

    def action(self):
        """Should return the best legal action from the current game"""
        raise NotImplementedError

    def evaluate(self):
        """Should return the legal (policy, outcome) result from the current game"""
        raise NotImplementedError