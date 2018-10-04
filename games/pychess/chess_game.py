import numpy as np
import chess

from ..wrappers import Game
from .utils import moves_dict, map_board

__all__ = [
    'Chess'
]

class Chess(Game):
    """
    A chess game. Initialise the chess game with the starting position given by fen.

    Params
    ------

    fen: string, default = rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        Starting FEN position of the game.
    """

    def __init__(
        self,
        fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        ):
        
        self.start_position = fen
        self.Board = chess.Board(fen=fen)
        move_id, id_move = moves_dict()
        self.move_id = move_id
        self.id_move = id_move

    def board(self):
        return str(self.Board)

    def get_board_size(self, nnet=False):
        if nnet:
            return (8, 8, 12)
        else:
            return (8, 8)

    def start(self):
        return self.start_position
    
    def current_player(self):
        turn = self.Board.turn
        if turn is True:
            # white
            return 1
        # black
        return -1
        
    def state(self):
        return self.Board.fen()

    def canonical_state(self):
        fen = self.Board.fen()
        out = fen.split(' ')[:-2]
        out = ' '.join(out)
        return out

    def set_state(self, fen):
        self.Board.set_fen(fen)

    def move(self, action):
        """Moves a piece. Accepts a move id that is converted to a uci move.
        Also accepts an 'undo' which undoes the last 2 moves."""
        if isinstance(action, str):
            action = action.lower()
            if action == 'undo':
                if len(self.Board.move_stack >= 2):
                    self.Board.pop()
                    self.Board.pop()
            else:
                # convert it to an ID and move that
                action = self.move_id.get(action, None)
                if action == None:
                    raise ValueError('The action %s is not a valid action' %action)
                self.move(action)
            return
        
        action = int(action)
        uci_move = self.id_move.get(action, None)
        if uci_move is None:
            raise ValueError('Given action %d does not have an associated move.' %action)
        
        move = chess.Move.from_uci(uci_move)
        if self.Board.is_legal(move):
            self.Board.push(move)
        else:
            print("Invalid move " + str(move))
            print(self.legal_moves_uci())
            print(self.Board)

    def action_size(self):
        return len(self.move_id)

    def legal_moves(self):
        """Returns the legal moves of the current game as a masking array."""
        moves = self.Board.legal_moves
        moves = [move.uci() for move in moves]
        moves = list(map(lambda move: self.move_id[move], moves))

        out = np.zeros(self.action_size())
        out[moves] = 1
        return out

    def legal_moves_uci(self):
        """Returns a list of the legal moves in uci format."""
        moves = self.Board.legal_moves
        moves = [move.uci() for move in moves]
        return moves

    def legal_moves_id(self):
        """Returns a list of the legal moves as their unique ids"""
        moves = self.Board.legal_moves
        moves = [move.uci() for move in moves]
        moves = list(map(lambda move: self.move_id[move], moves))
        return moves

    def winner(self):
        result = self.Board.result(claim_draw=True)
        if result == '*':
            # game still going
            return 0
        if result == '1-0' or result == '0-1':
            # it's always the turn of the loser at this point
            # so return the win value (for the player above us)
            return -1
        if result == '1/2-1/2':
            return 0.00001

    def represent_nn(self, state=None):
        """
        Takes a chess game and returns a matrix that can be used as an input for 
        neural networks. Takes advantage of the fact that the python-chess package
        already has a convenient string representation of a board. 

        This representation is somewhat similar to the one used by AlphaZero, only
        instead of having seperate planes for black and white pieces, we have a +1
        value for white and -1 value for black.

        Params
        ------

        state: Chess state, default=None
            The state that is being played. If None, uses the current board state.

        Returns
        -------

        nn_input: array(8 x 8 x (6 + 6))
            The input matrix to a neural network. Contains the 8x8 board in (6 + 6) planes,
            6 planes for the positions of the pieces (p, k, b, r, q, k), and 6 planes of 
            constant value denoting the color of the current player, total move count, p1
            and p2 castling rights, enpassaint rights, and a no-progress count. Is always from the POV of white.
        """
        WHITE = 1 # from the pychess definition
        BLACK = 0 
        if state is not None:
            board_state = self.state()
            self.set_state(state)

        # define the board matrix
        board = self.Board
        board_str = str(board)
        board_str = board_str.split('\n')
        board_str = [row.split(' ') for row in board_str]
        board_nn = np.array(board_str)
        board_nn = map_board(board_nn)

        # define the constant value matrix
        ones = np.ones((8, 8, 1))
        color = ones * self.current_player()
        move_count = ones * len(board.move_stack)
        p1_castle = ones * (board.has_kingside_castling_rights(WHITE) - 2 * board.has_queenside_castling_rights(WHITE))
        p2_castle = ones * (board.has_kingside_castling_rights(BLACK) - 2 * board.has_queenside_castling_rights(BLACK))
        if board.has_legal_en_passant():
            enpassant = ones * board.ep_square
        else:
            enpassant = ones * 0.
        no_progress = ones * (board.halfmove_clock / 2)
        value_nn = np.concatenate([color, move_count, p1_castle, p2_castle, enpassant, no_progress], axis=2)

        nn_input = np.concatenate([board_nn, value_nn], axis=2)
        
        if state is not None:
            # reset state
            self.set_state(board_state)
        return nn_input
    