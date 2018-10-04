class NNetWrap(object):
    """Base class for neural networks. 
    All neural nets should derive from this class.
    """

    def __init__(self):
        pass

    def build(self):
        """Handles the construction of the neural net."""
        raise NotImplementedError
    
    def train(self):
        """Handles the training of the neural net."""
        raise NotImplementedError

    def predict(self):
        """Handles making predictions."""
        raise NotImplementedError

    def save_model(self):
        """Handles saving a model."""
        raise NotImplementedError

    def load_model(self):
        """Handles loading a model."""
        raise NotImplementedError