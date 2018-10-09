import numpy as np

class TicTacToeEngine(object):
    """ A class controlling a TicTacToe game."""

    def __init__(self, n):
        """Create a tictactoe game.

        Params
        ------

        n: int
            The size of the n x n board.
        """
        self.n = int(n)
        self.size = (self.n, self.n)
        self.board = np.zeros(self.size, dtype=int)
        self.player = 1 # 1 for o's, -1 for x's, o always goes first

    def legal_moves(self):
        """Returns the legal moves."""
        moves = []
        if self.result() == '*':
            moves = [(i, j) for i in range(self.n) for j in range(self.n) if self.board[i, j] == 0]
        return moves

    def move(self, coord):
        """Places an x or o at the coordinate given. Input should be a tuple of the form (row, col) 
        where row indicates which row to play in and col idicates which col to play in."""
        legal = self.legal_moves()
        if not coord in legal:
            raise ValueError('Given coord %s is not legal' %str(coord))
        row, col = coord
        self.board[row, col] = self.player
        self.player = -self.player

    def result(self):
        """Checks the result of the game, '*' if still playing, 
        '1-0' for o win, '0-1' for x win, '1/2-1/2' for draw."""
        # check rows
        for i in range(self.n):
            if np.all(self.board[i] == 1):
                return '1-0'
            if np.all(self.board[i] == -1):
                return '0-1'

        # check cols
        for j in range(self.n):
            if np.all(self.board[:, j] == 1):
                return '1-0'
            if np.all(self.board[:, j] == -1):
                return '0-1'
                
        # check main diagonal
        if np.all(np.diag(self.board) == 1):
            return '1-0'
        if np.all(np.diag(self.board) == -1):
            return '0-1'

        # check anti-diagonal
        if np.all(np.diag(np.fliplr(self.board)) == 1):
            return '1-0'
        if np.all(np.diag(np.fliplr(self.board)) == -1):
            return '0-1'

        # have checks for all winning conditions, check for a draw
        legal = [(i, j) for i in range(self.n) for j in range(self.n) if self.board[i, j] == 0]
        if len(legal) == 0:
            return '1/2-1/2'
        
        # game must still be going on
        return '*'

    def set_board(self, board):
        """Accepts a string state representing the board or a numpy array.
        Also checks if the board is a valid tictactoe game."""
        if isinstance(board, np.ndarray):
            if not is_valid(board):
                raise ValueError('Given board is not a valid tictactoe board.')
            self.board = board
            self.n = board.shape[0]
            self.size = board.shape
            self.player = get_player(board)
        elif isinstance(board, bytes):
            board = np.fromstring(board, int)
            n = np.sqrt(len(board))
            if int(n) != n:
                raise ValueError('Given board is not square')
            n = int(n)
            board = board.reshape((n,n))
            self.set_board(board)
        else:
            raise ValueError('Misunderstood board type.')

    def board_state(self):
        """Returns a string representation of the board. Not for displaying."""
        return self.board.tostring()

    def display(self):
        """Returns a pretty string representation of the board."""
        out = ''
        # convert the board to x's and o's
        convert = lambda x: 'o' if x==1 else 'x' if x==-1 else '.'
        # create the header row
        header = [' ' if i == 0 else str(i-1) for i in range(self.n+1)]
        header.append('\n')
        out += ' '.join(header)
        # convert rows
        for i in range(self.n):
            row = self.board[i].tolist()
            row = list(map(convert, row))
            row.insert(0, str(i))
            row.append('\n')
            out += ' '.join(row)
        out = out[:-1] # all but the last \n
        return out

    def symmetries(self):
        board = self.board
        rots = [board, np.rot90(board), np.rot90(board, 2), np.rot90(board, 3)]
        flips = list(map(np.fliplr, rots)) + list(map(np.flipud, rots))
        syms = rots + flips
        syms = list(map(lambda x: x.tostring(), syms))
        syms = list(set(syms))
        return syms
            

def is_valid(board):
        """Checks if the given board is valid."""
        board_sum = np.sum(board)
        # this is either 1 or 0
        n, m = board.shape
        if n != m:
            return False
        if board_sum == 1 or board_sum == 0:
            return True
        else:
            return False

def get_player(board):
    """Returns the current player given a board. Assumes the board is valid and is an array."""
    board_sum = np.sum(board)
    if board_sum == 1:
        # black's turn
        return -1
    if board_sum == 0:
        # white's turn
        return 1