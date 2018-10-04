import datetime
import numpy as np

class MCTS(object):
    """
    Class controlling a basic Monte Carlo Tree Search Algorithm
    """
    def __init__(self, game, **kwargs):
        self.game = game
        self.exploration = kwargs.get('exploration', np.sqrt(2)) #exploration parameter
        self.train_time = kwargs.get('train_time', 0) # training time, 0 for infinite time (use episodes)
        self.calculation_time = datetime.timedelta(seconds=self.train_time)
        self.episodes = kwargs.get('episodes', 100) # max number of episodes
        self.max_moves = kwargs.get('max_moves', 1000)
        self.action_space = np.arange(self.game.action_size())
        self.verbose = kwargs.get('verbose', 0)
        # set up the mcts
        self.Qsa = {} # quality of move given an action, the average of v over Nsa outcomes
        self.Nsa = {} # the number times this edge was visited
        self.visited_states = set()
        self.trained = False
        # store the root node of the mcts
        self.game_start = self.game.state()

    def update(self):
        # updates the mcts after a move has been made
        # essentially just changes the starting position
        self.game_start = self.game.state()
        self.trained = False

    def reset(self):
        """Reset the mcts."""
        self.game.set_state(self.game_start)
        self.Qsa = {}
        self.Nsa = {}
        self.visited_states = set()
        self.trained = False

    def get_Qsa(self, s, act):
        a = np.asarray(act)

        if a.size == 1:
            if (s, act) not in self.Qsa:
                return 0
            return self.Qsa[(s, act)]

        out = np.zeros(a.size)
        for i in range(a.size):
            if (s, a[i]) not in self.Qsa:
                out[i] = 0
            else:
                out[i] = self.Qsa[(s, a[i])]
        if out.size == 1:
            return out[0]
        return out

    def get_Nsa(self, s, act):
        a = np.asarray(act)

        if a.size == 1:
            if (s, act) not in self.Nsa:
                return 0
            return self.Nsa[(s, act)]
        
        out = np.zeros(a.size)
        for i in range(a.size):
            if (s, a[i]) not in self.Nsa:
                out[i] = 0
            else:
                out[i] = self.Nsa[(s, a[i])]
        return out

    def Ns(self, s):
        return sum(self.get_Nsa(s, self.action_space))

    def pprint(self, msg):
        if self.verbose:
            print(msg)

    def train_episode(self):
        """Trains a single episode of MCTS."""
        assert self.game.state() == self.game_start
        self.build()
        self.game.set_state(self.game_start)

    def train(self):
        """Trains the MCTS for either the given amount of time (if defined) or the number of episodes."""
        if self.train_time <= 0:
            # train on episodes alone
            time = []
            b = datetime.datetime.utcnow()
            for ep in range(self.episodes+1):
                begin = datetime.datetime.utcnow()
                self.train_episode()
                end = datetime.datetime.utcnow()

                if self.verbose:
                    delta = end - begin
                    delta = delta.total_seconds()
                    time.append(delta)
                    eta = np.mean(time)*(self.episodes - ep)
                    print("Ep %d done in %.3f seconds, ETA %.2f seconds" %(ep+1, delta, eta))
                
            e = datetime.datetime.utcnow()
            delta = (e-b).total_seconds()
            self.pprint("Done training, took %.2f seconds" %(delta))
        else:
            begin = datetime.datetime.utcnow()
            ep = 0
            while datetime.datetime.utcnow() - begin < self.calculation_time:
                ep_begin = datetime.datetime.utcnow()
                self.train_episode()
                ep_end = datetime.datetime.utcnow()

                ep += 1
                if self.verbose:
                    delta = ep_end - ep_begin
                    delta = delta.total_seconds()
                    total = datetime.datetime.utcnow() - begin
                    total = total.total_seconds()
                    print("Ep %d done in %.3f seconds, TOTAL: %.2f" %(ep, delta, total))

            self.episodes = ep
            end = datetime.datetime.utcnow()
            delta = end - begin
            delta = delta.total_seconds()
            self.pprint("Done training, took %.2f seconds" %(delta))  

        self.trained = True

    def simulate(self, s=None):
        """
        Simulates a game from the state s. Returns the winner value and the player at the time.
        Currently, the game engine will always return a winner value of -1, and will always do this on 
        the turn of the loser.
        """
        if s == None:
            s = self.game.state()
        else:
            self.game.set_state(s)

        while self.game.winner() == 0:
            legal_moves = self.game.legal_moves()
            legal_moves /= sum(legal_moves)
            a = np.random.choice(self.action_space, p=legal_moves)
            self.game.move(a)
        return self.game.winner(), self.game.current_player()

    def select(self, s=None):
        """Returns an action that maximises a UCT based search"""
        if s == None:
            s = self.game.state()
        else:
            self.game.set_state(s)

        legal_moves = self.game.legal_moves()

        uct = self.get_Qsa(s, self.action_space) + self.exploration * np.sqrt(self.Ns(s))/(1+self.get_Nsa(s, self.action_space))

        # make sure we don't pick an illegal move
        illegal_moves = 1 - legal_moves
        uct = uct - illegal_moves * 1e6

        a = np.argmax(uct)

        if len(np.argwhere(uct[a] == uct)) > 0:
            # we have more than one argmax
            # randomly choose between them
            a_args = np.argwhere(uct[a] == uct)
            a = np.random.choice(a_args.reshape(len(a_args)))
        
        return a

    def build(self, num_moves=0):
        """
        Builds the MCTS tree using the game engine.
        This performs one episode of an MCTS training cycle
        """

        s = self.game.state()
        p = self.game.current_player() # player at this node

        # if terminal, return value
        if self.game.winner() != 0:
            return self.game.winner(), self.game.current_player()

        # if this is a new child node, we need to simulate it's outcome
        if s not in self.visited_states:
            self.visited_states.add(s)
            v, player = self.simulate()
            self.game.set_state(s) # reset the board after a simulation
            return v, player

        a = self.select() # select an action
        self.game.move(a) # sets the new state of the game

        v, player = self.build(num_moves+1) # builds from this new state

        # the way our game engine works, v is always -1,
        # and the player will be the losing player, so...

        if p != player:
            # we're one of the winning players, so make v=1
            # positive v adds to our Qsa, meaning we chose a good action here
            v_node = -v
        else:
            # else we're one of the losing players, keep v = -1
            v_node = v

        # backpropagate
        self.Qsa[(s, a)] = (self.get_Nsa(s, a) * self.get_Qsa(s, a) + v_node)/(1 + self.get_Nsa(s, a))
        self.Nsa[(s, a)] = self.get_Nsa(s, a) + 1

        # return so that the nodes in the recursive path retrieve the same information
        return v, player


    def get_policy(self, prop=1, state=None):
        """
        This function will return the MCTS policy vector, either proportionally
        or greedily with respect to the visit counts at the root state.

        Params
        ------

        prop: float, default=1
            Determines the proportionality of the visitor counts

        state: type of game.state(), default=None
            Determines the policy from this state. If None, uses the current state.

        Returns
        -------

        policy: array
            A vector where the probability of the ith action is proportional to Nsa[(s,a)]**(1./prop)
        """
        if self.trained is False:
            self.pprint('Training untrained MCTS')
            self.train()

        s = self.game.state() if state == None else state
        counts = self.get_Nsa(s, self.action_space)

        if prop==0:
            best_action = np.argmax(counts)
            policy = np.zeros(self.game.action_size())
            policy[best_action] = 1
            return policy

        counts = counts**(1./prop)
        policy = counts / float(np.sum(counts))
        return policy



class MCTSNet(MCTS):
    """
    Class controlling the Monte Carlo Tree Search Algorithm with a neural net
    """

    def __init__(self, game, player, **kwargs):
        super(MCTSNet, self).__init__(game, **kwargs)
        self.player = player
        self.Ps = {}

    def reset(self):
        """Reset the mcts."""
        super(MCTSNet, self).reset()
        self.Ps = {}

    def simulate(self):
        policy, outcome = self.player.evaluate()
        legal_moves = self.game.legal_moves()
        sum_p = np.round(np.sum(policy), 8)
        if not sum_p == 1.:
            raise ValueError('Player generated policy is not normalized, got sum(pi)=' + str(np.sum(policy)))
        if (policy * legal_moves != policy).any():
            raise ValueError('Player generated policy includes non legal moves!')
        return policy, outcome

    def select(self, s=None):
        if s == None:
            s = self.game.state()
        else:
            self.game.set_state(s)

        legal_moves = self.game.legal_moves()

        puct = self.get_Qsa(s, self.action_space) + self.exploration * self.Ps[s] * np.sqrt(self.Ns(s)) / (1 + self.get_Nsa(s, self.action_space))

        # make sure we don't pick an illegal move
        illegal_moves = 1 - legal_moves
        puct = puct - illegal_moves * 1e6

        a = np.argmax(puct)

        if len(np.argwhere(puct[a] == puct)) > 0:
            # we have more than one argmax
            # randomly choose between them
            a_args = np.argwhere(puct[a] == puct)
            a = np.random.choice(a_args.reshape(len(a_args)))
        
        return a

    def build(self, num_moves=0):
        """
        Performs an iteration of MCTS. Updates the tree structure on every iteration. 
        Is called recursively.
        """

        # get the state of the game
        s = self.game.state()
        p = self.game.current_player() # player at this node
        legal_moves = self.game.legal_moves()

        # if terminal, return value
        if self.game.winner() != 0:
            return self.game.winner(), p

        # if this is a new child node, we need to simulate it's outcome
        if s not in self.visited_states:
            self.visited_states.add(s)
            policy, outcome = self.simulate()
            self.Ps[s] = policy
            return outcome, p

        a = self.select()
        self.game.move(a) # update to the next state

        if num_moves < self.max_moves:
            v, player = self.build(num_moves+1)
        else:
            # we've reached the maximum number of moves, assume a draw
            v = 1e-5
            player = 0

        if p != player:
            # we are not the player that was playing this value judgement was made
            # so make v_node = -v
            v_node = -v
        else:
            v_node = v

        # back propagate
        self.Qsa[(s,a)] = (self.get_Nsa(s,a) * self.get_Qsa(s,a) + v_node)/(self.get_Nsa(s,a) + 1)
        self.Nsa[(s,a)] = self.get_Nsa(s,a) + 1

        # return so that the nodes in the recursive path retrieve the same information
        return v, player
