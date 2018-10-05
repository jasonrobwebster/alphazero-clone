import numpy as np

def get_data(mcts, max_moves=150, nnet=True, prop_thresh=30, verbose=0, return_moves=True):
    """
    Gets the data that can be used to train an agent (typically a neural net). Uses MCTS
    to generate policies and rewards that are then returned.

    Params
    ------
    
    mcts: MCTS
        The monte carlo tree search object to use.

    max_moves: int, default=150
        The maximum number of moves the game can make before considering the game a draw.

    nnet: bool, default=False
        Whether the data should e acceptable to a neural network.

    prop_thresh: int, default=50
        Proportionality threshold for the prop constant in the mcts policy. The threshold defines the move
        after which the MCTS starts behaving greedily.

    verbose: int, default=0
        The verbosity of the state. Accepts 0 or 1 (verbose or not).

    return_move: bool, default=False
        Whether to return the number of moves as well as the data.

    Returns
    -------

    data: list
        The encoded data, containing training examples in the form [(state, target_pi, target_value),...].

    moves: int, optional
        The number of moves played. Only returns if return_move is True.
    """
    memory = [] # place to store states as we play
    possible_moves = mcts.action_space
    game= mcts.game
    board_state = game.state()

    for move in range(max_moves):
        # get the game state and current player for the nn
        states = game.get_symmetries(nnet)
        cur_play = game.current_player()

        # use mcts to get a policy
        prop = int(move < prop_thresh)
        mcts.train()
        policy = mcts.get_policy(prop=prop)

        # choose an action based off this state
        act = np.random.choice(possible_moves, p=policy)

        # store the state, policy, and player
        for state in states:
            memory.append([state, policy, cur_play])

        # perform this action
        #s = game.state()
        game.move(act)
        mcts.update()
        #print(mcts.get_Qsa(s, act), mcts.get_Nsa(s, act))
        #print(game.board())

        # check if the game is over
        v = game.winner()
        if v !=0:
            # game over state
            # it's currently the move of the loser, so v=-1
            # all states that have this player should have a v=-1
            # all states that have the other player should have v=1
            # so check the current player
            cur_play = game.current_player()
            # so if cur_play = sa.cur_play, return v=-1
            # if cur_play != sa.cur_play, return v=1
            data = [(x[0], x[1], v if x[2] == cur_play else -v) for x in memory]
            # reset the game
            game.set_state(board_state)
            mcts.update()
            if return_moves:
                return data, move+1
            return data
    # max moves was reached
    # here the outcome is a draw
    if verbose:
        print("Game ended in draw, max_moves was met")
    v = 0
    data = [(x[0], x[1], v) for x in memory]
    game.set_state(board_state)
    mcts.update()
    if return_moves:
        return data, move+1
    return data