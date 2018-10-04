import numpy as np

def moves_dict():
    """Generates all available moves and maps them to a unique id.
    Returns a tuple of dicts (move_id, id_move).

    Returns
    -------

    move_id: dict
        A dict giving the unique of a uci move.

    id_move: dict
        Inverse dict to move_id. Gives a move for a given ID.
    """
    # handle all normal moves a1a2, a1a3, ... h8h7
    moves = [
        chr(sq1) + str(i+1) + chr(sq2) + str(j+1)
        for sq1 in range(ord('a'), ord('h')+1) for i in range(8) for sq2 in range(ord('a'), ord('h')+1) for j in range(8)
        if (sq1==sq2 and i==j) is False
    ]

    # handle straight pawn promotions
    # e.g. a7a8r, h2h1q etc
    promotions = [
        chr(sq1) + str(i) + chr(sq2) + str(j) + p
        for sq1 in range(ord('a'), ord('h')+1) for i in [2, 7] for sq2 in range(ord('a'), ord('h')+1) for j in [1, 8]
        for p in ['b', 'n', 'r', 'q']
        if (sq1==sq2 and ((i==7 and j==8) or (i==2 and j==1)))
    ]

    # handle cross pawn promotions
    # a7b8, h2g1, etc
    cross = [
        chr(sq1) + str(i) + chr(sq2) + str(j) + p
        for sq1 in range(ord('a'), ord('h')+1) for i in [2, 7] for sq2 in range(ord('a'), ord('h')+1) for j in [1, 8]
        for p in ['b', 'n', 'r', 'q']
        if ((sq1==sq2-1 or sq1==sq2+1) and ((i==7 and j==8) or (i==2 and j==1)))
    ]

    moves = moves + promotions + cross
    # associate each move with a unique ID
    move_id = dict([(move, i) for i, move in enumerate(moves)])
    # get the reverse mapping
    id_move = dict([(i, move) for i, move in enumerate(moves)])

    return move_id, id_move

def map_board(board):
    """Takes a piece as a string (here either '.', 'p', 'n', ...) and converts it to
    a 6d vector."""
    # define dict mapping a piece to a index
    plane_map = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5
    }
    out = np.zeros([8, 8, 6])
    for i in range(8):
        for j in range(8):
            piece = board[i, j]
            if piece == '.':
                continue
            # white pieces are represented in uppercase, black pieces in lowercase
            if piece.upper() == piece:
                # It's a white piece
                color = 1
            else:
                # It's black
                color = -1
            k = plane_map[piece.lower()]
            out[i, j, k] = color
    return out


