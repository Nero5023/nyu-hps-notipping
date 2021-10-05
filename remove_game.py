import time

from game import NoTippingGame
from game import pos_2_idx
from game import idx_2_pos
from game import Player, bit_to_1d_array
import numpy as np
import random


class NoTippingRemoveGame:
    BoardState = []
    @staticmethod
    def INIT_State(states, state_bit):
        NoTippingRemoveGame.BoardState = states
        left = NoTippingGame.cal_left_torque(0, NoTippingGame.BoardWeight)
        right = NoTippingGame.cal_right_torque(0, NoTippingGame.BoardWeight)

        num_weight = 0
        for idx, weight in enumerate(state):
            if weight is None or weight == 0:
                continue
            num_weight += 1
            pos = idx_2_pos(idx)
            left += NoTippingGame.cal_left_torque(pos, weight)
            right += NoTippingGame.cal_right_torque(pos, weight)

        return NoTippingRemoveGame(state_bit, left, right)

    def __init__(self, state_bit, left, right):
        self.curr_left_torque = left
        self.curr_right_torque = right

        self.state_bit = state_bit

    @staticmethod
    def cal_left_torque(pos, weight):
        return weight * (NoTippingGame.LeftStandPos - pos)

    @staticmethod
    def cal_right_torque(pos, weight):
        return weight * (NoTippingGame.RightStandPos - pos)

    def take_move(self, idx):
        mask = np.uint64(0b1 << idx)
        new_bit = self.state_bit | mask
        pos = idx_2_pos(idx)
        weight = NoTippingRemoveGame.BoardState[idx]
        new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)
        return NoTippingRemoveGame(new_bit, new_left, new_right)

    def take_move_pos(self, pos):
        idx = pos_2_idx(pos)
        return self.take_move(idx)

    # def remove_inplace(self, pos):
    #     idx = pos_2_idx(pos)
    #     weight = self.state[idx]
    #     self.state[idx] = None
    #
    #     new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
    #     new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)
    #
    #     self.curr_left_torque = new_left
    #     self.curr_right_torque = new_right
    #     self.num_weight -= 1
    #
    #     mask = np.uint64(0b1 << idx)
    #     self.state_bit = self.state_bit | mask
    #
    # def undo(self, pos):
    #     idx = pos_2_idx(pos)
    #     weight = self.origin_state[idx]
    #     self.state[idx] = weight
    #
    #     new_left = self.curr_left_torque + NoTippingGame.cal_left_torque(pos, weight)
    #     new_right = self.curr_right_torque + NoTippingGame.cal_right_torque(pos, weight)
    #
    #     self.curr_left_torque = new_left
    #     self.curr_right_torque = new_right
    #     self.num_weight += 1
    #
    #     mask = ~np.uint64(0b1 << idx)
    #     self.state_bit = self.state_bit & mask

    @property
    def is_board_flip(self):
        return self.curr_left_torque > 0 or self.curr_right_torque < 0

    def will_success_remove(self, pos):
        idx = pos_2_idx(pos)
        weight = NoTippingRemoveGame.BoardState[idx]

        new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)
        return new_left <= 0 and new_right >= 0

    def get_legal_actions(self):
        return bit_to_1d_array(~self.state_bit, NoTippingGame.BoardLen+1)


class RemoveGameState:
    @staticmethod
    def INIT_State(state, state_bit):
        game = NoTippingRemoveGame.INIT_State(state, state_bit)
        return RemoveGameState(game, Player.BLACK)

    def __init__(self, game: NoTippingRemoveGame, to_play: Player):
        self.game = game
        self.to_play = to_play

    def take_move(self, idx):
        new_game = self.game.take_move(idx)
        return RemoveGameState(new_game, self.to_play.rival())

    def take_move_pos(self, pos):
        idx = pos_2_idx(pos)
        return self.take_move(idx)

    def get_legal_actions(self):
        actions = self.game.get_legal_actions()
        for idx, val in enumerate(actions):
            if val == 1:
                pos = idx_2_pos(idx)
                if not self.game.will_success_remove(pos):
                    actions[idx] = 0

        if sum(actions) == 0:
            return self.game.get_legal_actions()
        return actions

    @property
    def is_terminal(self):
        return self.game.is_board_flip or sum(self.get_legal_actions()) == 0

    def winner(self):
        if self.game.is_board_flip:
            return self.to_play
        return None

    @property
    def is_board_flip(self):
        return self.game.is_board_flip

    def to_play_factor(self):
        if self.to_play == Player.BLACK:
            return 1
        else:
            return -1

    def need_pass(self):
        return False

    def legal_success_pos(self):
        idxs = self.get_legal_actions()
        poss = []
        for idx, val in enumerate(idxs):
            if val == 1:
                poss.append(idx_2_pos(idx))
        return poss

    def legal_pos(self):
        idxs = self.get_legal_actions()
        poss = []
        for idx, val in enumerate(idxs):
            poss.append(idx_2_pos(idx))
        return poss

    def to_state(self):
        return self.game.state_bit


def Max(state: RemoveGameState, alpha, beta, cache):
    if state.is_board_flip:
        return 1
    if state.is_terminal:
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = -1
    for pos in state.legal_success_pos():
        newstate = state.take_move_pos(pos)
        val = Min(newstate, alpha, beta, cache)
        if val > best:
            best = val
        if best == 1:
            cache[state_id] = best
            return 1
        if val >= beta:
            return best
        if val > alpha:
            alpha = val
    cache[state_id] = best
    return best


def Min(state: RemoveGameState, alpha, beta, cache):
    if state.is_board_flip:
        return -1
    if state.is_terminal:
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = 1
    for pos in state.legal_success_pos():
        new_state = state.take_move_pos(pos)
        val = Max(new_state, alpha, beta, cache)
        if val < best:
            best = val
        if best == -1:
            cache[state_id] = best
            return best
        if val <= alpha:
            return best
        if val < beta:
            beta = val
    cache[state_id] = best
    return best


class RemovePlayer:
    def __init__(self, board, board_bit):
        self.state = RemoveGameState.INIT_State(board, board_bit)
        self.cache = {}

    def take_move(self, pos):
        self.state = self.state.take_move_pos(pos)

    def random_pick(self):
        poss = self.state.legal_success_pos()
        if len(poss) == 0:
            return random.choice(self.state.legal_pos())
        else:
            return random.choice(poss)

    def pick_move(self):
        if len(self.state.game.get_legal_actions()) >= 21:
            return self.random_pick()
        else:
            if self.state.to_play == Player.BLACK:
                val = Max(self.state, -1, 1, self.cache)
                if val <= -1:
                    return self.random_pick()
                target_val = val
                legal_poss = self.state.legal_success_pos()
                for pos in legal_poss:
                    new_state = self.state.take_move_pos(pos)
                    rival_val = Min(new_state, -1, 1, self.cache)
                    if rival_val == target_val:
                        return pos
            else:
                val = Min(self.state, -1, 1, self.cache)
                if val >= 1:
                    return self.random_pick()
                target_val = val
                legal_poss = self.state.legal_success_pos()
                for pos in legal_poss:
                    new_state = self.state.take_move_pos(pos)
                    rival_val = Max(new_state, -1, 1, self.cache)
                    if rival_val == target_val:
                        return pos




if __name__ == '__main__':
    # state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 10, 9, 8, 7, 6, 5, 4, 3, 3, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    # bit = np.uint64(2305842459458142207)
    # init_game = NoTippingRemoveGame(state, np.uint64(2305842459458142207))
    # state = RemoveGameState(init_game, Player.BLACK)
    # start = time.time()
    # print(Max(state, -1, 1, {}))
    # print(time.time()-start)

    # state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    # state = RemoveGameState.INIT_State(state, np.uint64(2305840810190503935))
    # start = time.time()
    # print(Max(state, -1, 1, {}))
    # print(time.time() - start)


    # state = [None, None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 2, None, 3, None, None, None, None, None, None, None, None, None, None, 2, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None]
    # bit = np.uint64(2303591071877169023)
    from game import GameState
    from game import PutPlayer
    k = 10
    game = GameState.INIT_State(k, 60)
    player = PutPlayer(k)

    positions = [None] * 61
    positions[pos_2_idx(-4)] = 3
    while not game.is_terminal():
        pos, weight = player.pick_move()
        print("{} {}".format(pos, weight))
        player.take_move(pos, weight)
        game = game.take(pos, weight)
        positions[pos_2_idx(pos)] = weight
    print(positions)
    print(game.game.board_available_move)
    print(game.winner())
    state = positions[:]
    bit = game.game.board_available_move

    state = RemoveGameState.INIT_State(state, bit)
    start = time.time()
    print(Max(state, -1, 1, {}))
    print(time.time() - start)