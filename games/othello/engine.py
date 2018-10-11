import numpy as np

class OthelloEngine(object):
    """Handles the Othello game"""

    def __init__(self, n):
        """Creat an n x n Othello game

        params
        ------

        n : int
            The size of the n x n board
        """
        if int(n) <= 2:
            raise ValueError('Invalid board size, must have n > 2. n = %d' %int(n))
        if int(n) % 2 != 0:
            raise ValueError('Must have an even board size, got n = %d' %int(n))
        self.n = int(n)
        self.size = (self.n, self.n)
        self.board = np.zeros(self.size, dtype=int)
        self.board[int(self.n/2) - 1, int(self.n/2) - 1] = -1
        self.board[int(self.n/2) - 1, int(self.n/2)] = 1
        self.board[int(self.n/2), int(self.n/2) - 1] = 1
        self.board[int(self.n/2), int(self.n/2)] = -1
        self.player = 1 # 1 for B, -1 for W, B always goes first
        #self.result = '*' # '*' if still playing, '1-0' for B win, '0-1' for W win, '1/2-1/2' for draw

    # utility functions
    def return_bound(self, x, y):
        """Returns the  coord (i, j) with the boundries of the board such that (i, j) is a result of a minmax operation on (x, y)"""
        i = min(max(x, 0), self.n-1)
        j = min(max(y, 0), self.n-1)
        return (i, j)

    def list_neighbours(self, x, y):
        """Checks the grid surrounding a coord (x, y) for stones of the opposite color. 
        Returns a list of positions of the opposing colors stones."""
        out = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                ii, jj = self.return_bound(x+i, y+j)
                if self.board[ii, jj] == -self.player:
                    out.append((ii, jj))
        return out

    def check_legal(self, x, y):
        if self.board[x, y] != 0:
            return False
        neighbours = self.list_neighbours(x, y)
        # check for a contiguous line of opossing stones in the direction of the neighbours
        for neighbour in neighbours:
            dx = x - neighbour[0]
            dy = y - neighbour[1]
            i = 1
            while x - i*dx >= 0 and x - i*dx < self.n and y - i*dy >= 0 and y - i*dy < self.n:
                if self.board[x - i*dx, y - i*dy] == 0:
                    # empty board, this direction is not legal
                    break
                elif self.board[x - i*dx, y - i*dy] == -self.player:
                    # found another opposing stone, continue
                    i += 1
                    continue
                elif self.board[x - i*dx, y - i*dy] == self.player:
                    # found a friendly stone after a contiguous line of opposing stones
                    return True
        # nothing was found
        return False

    # relevant functions
    def legal_moves(self):
        """Returns the legal moves for the player"""
        moves = [(x, y) for x in range(self.n) for y in range(self.n) if self.check_legal(x, y)]
        return moves

    def move(self, coord):
        """Updates the board given a move to a coordinate"""
        if coord == (self.n, self.n):
            self.player = -self.player
            return
        
        legal = self.legal_moves()
        
        if not coord in legal:
            raise ValueError('Given coord %s is not legal' %str(coord))

        x, y = coord
        neighbours = self.list_neighbours(x, y)

        # figure out what stones must be flipped
        lines = []
        for neighbour in neighbours:
            line = []
            dx = x - neighbour[0]
            dy = y - neighbour[1]
            i = 1
            while x - i*dx >= 0 and x - i*dx < self.n and y - i*dy >= 0 and y - i*dy < self.n:
                if self.board[x - i*dx, y - i*dy] == 0:
                    # empty position, this direction is not legal
                    break
                elif self.board[x - i*dx, y - i*dy] == -self.player:
                    # found another opposing stone, continue
                    line.append((x - i*dx, y - i*dy))
                    i += 1
                    continue
                elif self.board[x - i*dx, y - i*dy] == self.player:
                    # found a friendly stone after a contiguous line of opposing stones
                    lines.append(line)
                    break

        # update the board state
        self.board[x, y] = self.player
        for line in lines:
            for coord in line:
                i, j = coord
                self.board[i, j] = self.player

        # change the player
        self.player = -self.player

    def result(self):
        """Returns the result of the game, '*' if still playing, 
        '1-0' for B win, '0-1' for W win, and '1/2-1/2' for draw."""
        # check for legal moves from the current player
        moves = self.legal_moves()
        # check for legal moves for the opposing player
        self.player = -self.player
        opp_moves = self.legal_moves()
        self.player = -self.player

        if len(opp_moves) == 0 and len(moves) == 0:
            # the game is over
            # find the difference in scores
            score = np.sum(self.board)
            if score > 0:
                # B wins
                # for consistency with other engines, we set the player to that of the loser
                self.player = -1
                return '1-0'
            if score < 0:
                # W wins
                self.player = 1 # set consistency
                return '0-1'
            if score == 0:
                return '1/2-1/2'
        else:
            # the game continues
            return '*'

    def set_board(self, state):
        """Accepts a string state representing the board or a numpy array."""
        board, player = state.split(';')
        board = board.split(',')
        board = np.asarray(board, dtype=int)
        n = np.sqrt(len(board))
        if int(n) != n:
            raise ValueError('Given board is not square')
        n = int(n)
        board = board.reshape((n,n))

        self.board = board
        self.n = n
        self.size = board.shape
        self.player = int(player)

    def board_state(self):
        """Returns a string representation of the board. Not for displaying."""
        board = ','.join([str(num) for row in self.board for num in row])
        player = str(self.player)
        return board + ';' + player

    def display(self):
        """Returns a pretty string representation of the board."""
        out = ''
        # convert the board to x's and o's
        convert = lambda x: 'B' if x==1 else 'W' if x==-1 else '.'
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
        syms = list(map(lambda x: ','.join([str(num) for row in self.board for num in row]), syms))
        syms = list(set(syms))
        states = list(map(lambda x: x + ';' + str(self.player), syms))
        return states