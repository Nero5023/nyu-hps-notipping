import math
import time

import numpy as np
from remove_game import RemoveGameState, NoTippingRemoveGame
from collections import defaultdict

C_PUCT = 2
NOISE_EPSILON = 0.25
NOISE_ALPHA = 0.17
FEATURE_NUM = 7

MAX_MOVE = NoTippingRemoveGame.MAX_MOVE

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTSNode:
    def __init__(self, state: RemoveGameState, move, parent=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.is_expanded = False
        self.is_game_root = False
        self.is_search_root = False
        self.__is_terminal = False
        self.children = {}
        self.parent = parent
        self.pi = np.zeros([MAX_MOVE], dtype=np.float32)
        # +1 for pass move
        self.child_priors = np.zeros([MAX_MOVE], dtype=np.float32)
        self.child_total_values = np.zeros([MAX_MOVE], dtype=np.float32)
        self.child_number_visits = np.zeros([MAX_MOVE], dtype=np.float32)
        self.height = 0
        if parent:
            self.height = parent.height + 1

    @property
    def is_terminal(self):
        if self.__is_terminal is True:
            return self.__is_terminal
        # TODO: add is_terminal API
        self.__is_terminal = self.state.is_terminal
        return self.__is_terminal

    @property
    def N(self):
        return self.parent.child_number_visits[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def W(self):
        return self.parent.child_total_values[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_total_values[self.move] = value

    @property
    def Q(self):
        if self.height == 0:
            return 0
        return self.parent.child_Q()[self.move]

    def child_Q(self):
        # return self.child_total_values / (1+self.child_number_visits)
        return self.child_total_values * self.state.to_play_factor / (
                    self.child_number_visits + (self.child_number_visits == 0))

    def child_U(self):
        # self.edge_P * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)
        return C_PUCT*math.sqrt(self.N) * (self.child_priors / (1 + self.child_number_visits))

    def child_U_inject_noise(self):
        epsilon = 1e-5
        legal_moves = self.state.get_legal_actions() + epsilon
        alphas = legal_moves * ([NOISE_ALPHA] * MAX_MOVE)
        noise = np.random.dirichlet(alphas)
        p_with_noise = self.child_priors*(1-NOISE_EPSILON) + noise*NOISE_EPSILON
        return C_PUCT*math.sqrt(self.N) * (p_with_noise / (1 + self.child_number_visits))

    def best_child(self):
        # if self.is_search_root:
        #     # for search root add noise
        #     return np.argmax(self.child_Q() + self.child_U_inject_noise() + 1000 * self.state.get_legal_actions())
        # else:
        #     # add this to prevent self.child_Q() + self.child_U() < 0, others is == 0, which cloud take illegal action
        #     return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())
        # TODO: where to add noise
        return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

    def select_leaf(self):
        node = self
        while node.is_expanded:
            if node.is_terminal:
                break
            action = node.best_child()
            node = node.maybe_add_child(action)
        return node

    def maybe_add_child(self, action):
        if action not in self.children:
            self.children[action] = MCTSNode(state=self.state.take_move(action), move=action, parent=self)
        return self.children[action]

    def expand(self, child_prior_probabilities):
        if self.is_expanded:
            return
        self.is_expanded = True
        # normalize
        priors = np.multiply(child_prior_probabilities, self.state.get_legal_actions())
        normalized = priors/np.sum(priors)

        self.child_priors = normalized

    def back_update(self, value):
        # TODO: Check minigo mcts.py line 210
        # value = 1, black win
        # value = -1, white win
        node = self
        # check if node is root
        while True:
            node.N += 1
            node.W += value

            if node.is_search_root:
                break
            node = node.parent

    def children_pi(self, temperature):
        # todo: check possible move
        # TODO: maybe overflow here
        # /Users/Nero/local_dev/nyu/ml/ml-proj/mcts.py:104: RuntimeWarning: overflow encountered in power
        #   probs = self.child_number_visits ** (1 / temperature)
        # /Users/Nero/local_dev/nyu/ml/ml-proj/mcts.py:111: RuntimeWarning: invalid value encountered in true_divide
        #   self.pi = probs/sum_probs
        if np.sum(self.child_number_visits) == 0 or self.state.need_pass():
            # TODO: if this return pass move
            # self.pi = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float)
            # self.pi[PASS_MOVE] = 1
            assert "Not reachable"
        else:
            # probs = self.child_number_visits ** (1 / temperature)
            # sum_probs = np.sum(probs)
            # self.pi = probs/sum_probs
            # TODO: check if this correct
            pi = softmax(1.0/temperature * np.log(self.child_number_visits + 1e-10))
            self.pi = pi
        return self.pi

    def to_features(self):
        return None


class SentinelNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_values = defaultdict(float)
        self.child_number_visits = defaultdict(float)
        self.height = -1


# def UCT_search(state, num_reads):
#     root = MCTSNode(state, move=None, parent=SentinelNode())
#     for _ in range(num_reads):
#         leaf = root.select_leaf()
#         child_priors, value_estimate = NeuralNetRandom.evaluate(leaf.state)
#         leaf.expand(child_priors)
#         leaf.back_update(value_estimate)
#     return np.argmax(root.child_number_visits)

class MCTS:
    def __init__(self, state, state_bit):
        # self.nn = nn
        sentinel_node = SentinelNode()
        # TODO: pass move :0 change
        self.root = MCTSNode(RemoveGameState.INIT_State(state, state_bit), 0, sentinel_node)
        self.root.is_game_root = True
        self.root.is_search_root = True
        self.current_node = self.root
        self.move_num = 0
        self.winner = None

    def search(self, num_sims):
        for _ in range(num_sims):
            leaf = self.current_node.select_leaf()
            if leaf.is_terminal:
                leaf.back_update(leaf.state.winner_score())
                continue
            #
            # WITH neural net
            # child_priors, value_estimate = self.nn.predict(leaf.to_features())
            # leaf.expand(child_priors)
            # leaf.back_update(value_estimate)

            # for random simulate
            leaf.expand(np.ones([MAX_MOVE], dtype=np.float32))
            score = leaf.state.simulate_winner_score()
            leaf.back_update(score)

    def search_in_time(self, times_in_seconds):
        start = time.time()
        while True:
            leaf = self.current_node.select_leaf()
            if leaf.is_terminal:
                leaf.back_update(leaf.state.winner_score())
                continue
            # for random simulate
            leaf.expand(np.ones([MAX_MOVE], dtype=np.float32))
            score = leaf.state.simulate_winner_score()
            leaf.back_update(score)
            duration = time.time() - start
            if duration > times_in_seconds:
                return duration


    def take_move(self, move):
        # pi = self.current_node.children_pi(self.temperature)
        # move = pi.argmax()
        self.current_node = self.current_node.maybe_add_child(move)
        self.current_node.is_search_root = True
        self.move_num += 1

        if self.current_node.is_terminal:
            # update last node of children's pi
            self.current_node.children_pi(self.temperature)
            print("Termail")
            print(self.current_node.state)
            print("WINNER: {}".format(self.current_node.state.winner()))
            self.winner = self.current_node.state.winner()

    def pick_move(self):
        pi = self.current_node.children_pi(self.temperature)
        move = pi.argmax()
        return move

    def normalize_with_legal_moves(self, child_priors, legal_moves):
        legal_probs = np.multiply(child_priors, legal_moves)
        return legal_probs/np.sum(legal_probs)

    @property
    def temperature(self):
        return 1
        # if self.move_num <= 10:
        #     return 1
        # else:
        #     return 0.95 ** (self.move_num - 10)

    @property
    def is_terminal(self):
        return self.current_node.is_terminal


class NeuralNetRandom:
    def __init__(self, total_move):
        self.total_possible_move = total_move

    def predict(self, features):
        value = np.random.random()*2 - 1
        return np.random.random([self.total_possible_move]), value
