import time

from game import NoTippingGame
from game import pos_2_idx
from game import idx_2_pos
from game import Player
import numpy as np
import random

max_time = 0

G_CACHE = {}

class NoTippingRemoveGameInPlace:
    def __init__(self, state, state_bit):
        self.state = state
        self.origin_state = state[:]
        self.curr_left_torque = 0
        self.curr_right_torque = 0

        self.state_bit = state_bit

        left = NoTippingGame.cal_left_torque(0, NoTippingGame.BoardWeight)
        right = NoTippingGame.cal_right_torque(0, NoTippingGame.BoardWeight)

        # init_pos = -4
        # init_weight = 3
        # left += NoTippingGame.cal_left_torque(init_pos, init_weight)
        # right += NoTippingGame.cal_right_torque(init_pos, init_weight)

        self.num_weight = 0
        for idx, weight in enumerate(self.state):
            if weight is None or weight == 0:
                continue
            self.num_weight += 1
            pos = idx_2_pos(idx)
            left += NoTippingGame.cal_left_torque(pos, weight)
            right += NoTippingGame.cal_right_torque(pos, weight)

        self.curr_left_torque = left
        self.curr_right_torque = right



    @staticmethod
    def cal_left_torque(pos, weight):
        return weight * (NoTippingGame.LeftStandPos - pos)

    @staticmethod
    def cal_right_torque(pos, weight):
        return weight * (NoTippingGame.RightStandPos - pos)

    def remove_inplace(self, pos):
        idx = pos_2_idx(pos)
        weight = self.state[idx]
        self.state[idx] = None

        new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)

        self.curr_left_torque = new_left
        self.curr_right_torque = new_right
        self.num_weight -= 1

        mask = np.uint64(0b1 << idx)
        self.state_bit = self.state_bit | mask

    def undo(self, pos):
        idx = pos_2_idx(pos)
        weight = self.origin_state[idx]
        self.state[idx] = weight

        new_left = self.curr_left_torque + NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque + NoTippingGame.cal_right_torque(pos, weight)

        self.curr_left_torque = new_left
        self.curr_right_torque = new_right
        self.num_weight += 1

        mask = ~np.uint64(0b1 << idx)
        self.state_bit = self.state_bit & mask

    def is_board_flip(self):
        return self.curr_left_torque > 0 or self.curr_right_torque < 0

    def will_success_remove(self, pos):
        idx = pos_2_idx(pos)
        weight = self.state[idx]

        new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)
        return new_left <= 0 and new_right >= 0


class RemoveGameStateInPlace:
    def __init__(self, game: NoTippingRemoveGameInPlace, to_play: Player):
        self.game = game
        self.to_play = to_play

    def take_in_place(self, pos):
        self.game.remove_inplace(pos)
        self.to_play = self.to_play.rival()

    def undo(self, pos):
        self.game.undo(pos)
        self.to_play = self.to_play.rival()

    def is_terminal(self):
        return self.game.is_board_flip() or self.game.num_weight == 0

    def is_board_flip(self):
        return self.game.is_board_flip()

    def winner(self):
        if self.is_board_flip:
            return self.to_play
        return None

    def to_play_factor(self):
        if self.to_play == Player.BLACK:
            return 1
        else:
            return -1

    def need_pass(self):
        return False

    def legal_pos(self):
        poss = []
        for idx, val in enumerate(self.game.state):
            if val is None or val == 0:
                continue
            pos = idx_2_pos(idx)
            poss.append(pos)
        return poss

    def legal_success_pos(self):
        poss = []
        for idx, val in enumerate(self.game.state):
            pos = idx_2_pos(idx)
            if val is None or val == 0 or not self.game.will_success_remove(pos):
                continue
            poss.append(pos)
        # print(poss)
        return poss

    def to_state(self):
        return self.game.state_bit


def Max(state: RemoveGameStateInPlace, alpha, beta, cache):
    global max_time
    max_time += 1
    if state.is_board_flip():
        return 1
    if state.is_terminal():
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = -1
    for pos in state.legal_success_pos():
        state.take_in_place(pos)
        val = Min(state, alpha, beta, cache)
        state.undo(pos)
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
    # print(best)
    return best


def Min(state: RemoveGameStateInPlace, alpha, beta, cache):
    if state.is_board_flip():
        return -1
    if state.is_terminal():
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = 1
    for pos in state.legal_success_pos():
        state.take_in_place(pos)
        val = Max(state, alpha, beta, cache)
        state.undo(pos)
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
        init_game = NoTippingRemoveGameInPlace(board, board_bit)
        self.state = RemoveGameStateInPlace(init_game, Player.BLACK)
        self.cache = {}

    def take_move(self, pos):
        self.state.take_in_place(pos)

    def random_pick(self):
        poss = self.state.legal_success_pos()
        if len(poss) == 0:
            return random.choice(self.state.legal_pos())
        else:
            return random.choice(poss)

    def pick_move(self):
        if self.state.game.num_weight >= 21:
            return self.random_pick()
        else:
            if self.state.to_play == Player.BLACK:
                val = Max(self.state, -1, 1, self.cache)
                if val <= -1:
                    return self.random_pick()
                target_val = val
                legal_poss = self.state.legal_success_pos()
                for pos in legal_poss:
                    self.state.take_in_place(pos)
                    rival_val = Min(self.state, -1, 1, self.cache)
                    self.state.undo(pos)
                    if rival_val == target_val:
                        return pos
            else:
                val = Min(self.state, -1, 1, self.cache)
                if val >= 1:
                    return self.random_pick()
                target_val = val
                legal_poss = self.state.legal_success_pos()
                for pos in legal_poss:
                    self.state.take_in_place(pos)
                    rival_val = Max(self.state, -1, 1, self.cache)
                    self.state.undo(pos)
                    if rival_val == target_val:
                        return pos


class Simulation:
    def __init__(self, state, bit, to_play):
        init_game = NoTippingRemoveGameInPlace(state, bit)
        state = RemoveGameStateInPlace(init_game, to_play)
        self.state = state

    def random_select_move(self):
        poss = self.state.legal_success_pos()
        if len(poss) == 0:
            return random.choice(self.state.legal_pos())
        else:
            return random.choice(poss)

    def simulation_winner_score(self):
        # remain_weis = sum(self.state.game.get_legal_actions())
        remain_weis = self.state.game.num_weight
        while remain_weis > 14 and not self.state.is_terminal():
            move = self.random_select_move()
            self.state.take_in_place(move)

        if self.state.to_play == Player.BLACK:
            val = Max(self.state, -1, 1, G_CACHE)
            return val
        else:
            val = Min(self.state, -1, 1, G_CACHE)
            return val



if __name__ == '__main__':
    # state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 10, 9, 8, 7, 6, 5, 4, 3, 3, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    # init_game = NoTippingRemoveGame(state, np.uint64(2305842459458142207))
    # state = RemoveGameState(init_game, Player.BLACK)
    # start = time.time()
    # print(Max(state, -1, 1, {}))
    # print(time.time()-start)

    # state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

    from game import GameState
    from game import PutPlayer

    k = 8
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

    init_game = NoTippingRemoveGameInPlace(state, bit)
    state = RemoveGameStateInPlace(init_game, Player.BLACK)
    print("--------")
    start = time.time()
    print(Max(state, -1, 1, {}))
    print(time.time() - start)

    print(max_time)