from enum import Enum
import numpy as np
import math
import random
from copy import deepcopy
import time


class Player(Enum):
    BLACK = 0
    WHITE = 1

    def rival(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK

    def self(self):
        if self == Player.BLACK:
            return Player.BLACK
        else:
            return Player.WHITE


def pos_2_idx(pos):
    mid = NoTippingGame.BoardLen//2
    return pos + mid


def idx_2_pos(idx):
    mid = NoTippingGame.BoardLen//2
    return idx - mid


def bit_to_1d_array(bit, size):
    return np.array(list(reversed((("0" * size) + bin(bit)[2:])[-size:])), dtype=np.uint8)


class NoTippingGame:
    BoardLen = 60
    BoardWeight = 3
    MaxWeight = 10
    LeftStandPos = -3
    RightStandPos = -1

    @staticmethod
    def INIT_State(max_weight_block, board_len=60):
        NoTippingGame.MaxWeight = max_weight_block
        NoTippingGame.BoardLen = board_len
        left = NoTippingGame.cal_left_torque(0, NoTippingGame.BoardWeight)
        right = NoTippingGame.cal_right_torque(0, NoTippingGame.BoardWeight)

        init_pos = -4
        init_idx = pos_2_idx(init_pos)
        init_weight = 3
        left += NoTippingGame.cal_left_torque(init_pos, init_weight)
        right += NoTippingGame.cal_right_torque(init_pos, init_weight)

        init_board = np.uint64((0b1 << (NoTippingGame.BoardLen+1))-1)
        mask = ~np.uint64(0b1 << init_idx)
        init_board = init_board & mask

        black_state = np.uint64((0b1 << NoTippingGame.MaxWeight)-1)
        white_state = np.uint64((0b1 << NoTippingGame.MaxWeight)-1)

        game = NoTippingGame(init_board, left, right, black_state, white_state)
        return game

    def __init__(self, board_move_bit, left_torque, right_torque, black_state, white_state):
        self.curr_left_torque = left_torque
        self.curr_right_torque = right_torque
        self.black_state = black_state
        self.white_state = white_state
        self.board_available_move = board_move_bit
        # self.board_state = []

    @staticmethod
    def cal_left_torque(pos, weight):
        return weight * (NoTippingGame.LeftStandPos - pos)

    @staticmethod
    def cal_right_torque(pos, weight):
        return weight * (NoTippingGame.RightStandPos - pos)

    def __str__(self):
        return "left: {} right: {}".format(self.curr_left_torque, self.curr_right_torque)

    def get_legal_idxs(self):
        # idxs = []
        # for i in range(NoTippingGame.BoardLen+1):
        #     mask = np.uint64(0b1 << i)
        #     if mask & self.board_available_move:
        #         idxs.append(i)
        # return idxs
        return bit_to_1d_array(self.board_available_move, NoTippingGame.BoardLen+1)

    def bit_state(self, player: Player):
        if player == Player.BLACK:
            return self.black_state
        else:
            return self.white_state

    def get_legal_weights(self, player: Player):
        return bit_to_1d_array(self.bit_state(player), NoTippingGame.MaxWeight)

    def is_board_flip(self):
        return self.curr_left_torque > 0 or self.curr_right_torque < 0

    def check_if_move_will_success(self, pos, weight):
        # self_state = self.black_state
        # rival_state = self.white_state
        # if player == Player.WHITE:
        #     self_state = self.white_state
        #     rival_state = self.black_state
        new_left = self.curr_left_torque + NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque + NoTippingGame.cal_right_torque(pos, weight)
        return new_left <= 0 and new_right >= 0

    def take_move(self, player, pos, weight):
        self_state = self.black_state
        rival_state = self.white_state
        if player == Player.WHITE:
            self_state = self.white_state
            rival_state = self.black_state
        new_left = self.curr_left_torque + NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque + NoTippingGame.cal_right_torque(pos, weight)
        idx = pos_2_idx(pos)
        mask = ~np.uint64(0b1 << idx)
        new_board = self.board_available_move & mask

        weight_mask = ~np.uint64(0b1 << (weight-1))
        self_state = self_state & weight_mask

        if player == Player.BLACK:
            game = NoTippingGame(new_board, new_left, new_right, self_state, rival_state)
            return game
        else:
            game = NoTippingGame(new_board, new_left, new_right, rival_state, self_state)
            return game

    def remove(self, player, pos):
        idx = pos_2_idx(pos)
        self_state = self.black_state
        rival_state = self.white_state
        if player == Player.WHITE:
            self_state = self.white_state
            rival_state = self.black_state
        new_left = self.curr_left_torque - NoTippingGame.cal_left_torque(pos, weight)
        new_right = self.curr_right_torque - NoTippingGame.cal_right_torque(pos, weight)
        mask = np.uint64(0b1 << idx)
        new_board = self.board_available_move | mask

        weight_mask = np.uint64(0b1 << (weight - 1))
        self_state = self_state | weight_mask

        if player == Player.BLACK:
            game = NoTippingGame(new_board, new_left, new_right, self_state, rival_state)
            return game
        else:
            game = NoTippingGame(new_board, new_left, new_right, rival_state, self_state)
            return game


class GameState:
    @staticmethod
    def INIT_State(max_weight_block, board_len=60):
        nt_game = NoTippingGame.INIT_State(max_weight_block, board_len)
        return GameState(nt_game, Player.BLACK)

    def __init__(self, game: NoTippingGame, to_play: Player):
        self.game = game
        self.to_play = to_play

    def take(self, pos, weight):
        new_nt = self.game.take_move(self.to_play, pos, weight)
        return GameState(new_nt, self.to_play.rival())

    def remove(self, pos):
        new_nt = self.game.remove(self.to_play, pos)
        return GameState(new_nt, self.to_play.rival())

    # check if board flip
    @property
    def is_board_flip(self):
        return self.game.is_board_flip()

    def is_terminal(self):
        return self.is_board_flip or self.is_no_weights()

    def is_no_weights(self):
        return self.game.bit_state(self.to_play) == 0

    def winner(self):
        if self.is_board_flip:
            return self.to_play
        return None

    @property
    def to_play_factor(self):
        if self.to_play == Player.BLACK:
            return 1
        else:
            return -1

    def need_pass(self):
        return False

    def __str__(self):
        return self.game.__str__()

    def self_rival_state(self):
        if self.to_play == Player.BLACK:
            return self.game.black_state, self.game.white_state
        else:
            return self.game.white_state, self.game.black_state

    def to_state(self):
        self_s, rival_s = self.self_rival_state()
        return self_s, rival_s, self.game.board_available_move, self.game.curr_left_torque, self.game.curr_right_torque

    def legal_poss(self):
        l_idxs = self.game.get_legal_idxs()
        res = []
        for idx, val in enumerate(l_idxs):
            if val == 0:
                continue
            res.append(idx_2_pos(idx))
        return res

    def self_legal_weight(self):
        array = self.game.get_legal_weights(self.to_play)
        return list(np.where(array == 1)[0] + 1)

    def if_success_put_at(self, pos, weight):
        return self.game.check_if_move_will_success(pos, weight)

    def get_legal_move_tuple(self):
        moves = []
        for pos in self.legal_poss():
            for weight in self.self_legal_weight():
                moves.append((pos, weight))
        return moves

    def get_legal_success_move(self):
        moves = []
        for pos in self.legal_poss():
            for weight in self.self_legal_weight():
                if self.if_success_put_at(pos, weight):
                    moves.append((pos, weight))
                # moves.append((pos, weight))
        # print(len(moves))
        return moves


def can_win(game_state: GameState, cache):
    tuple_state = game_state.to_state()
    if tuple_state in cache:
        return cache[tuple_state]
    if game_state.is_board_flip:
        return True

    for legal_pos in game_state.legal_poss():
        for weight in game_state.self_legal_weight():
            if game_state.if_success_put_at(legal_pos, weight):
                continue
            new_game_s = game_state.take(legal_pos, weight)
            rival_can_win = can_win(new_game_s, cache)
            if not rival_can_win:
                cache[tuple_state] = True
                return True
    cache[tuple_state] = False
    return False


def Max(state: GameState, alpha, beta, cache):
    if state.is_board_flip:
        return 1
    if state.is_no_weights():
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = float('-inf')
    for pos, weight in state.get_legal_success_move():
        new_state = state.take(pos, weight)
        val = Min(new_state, alpha, beta, cache)
        if val > best:
            best = val
        if val >= beta:
            cache[state_id] = best
            return best
        if val > alpha:
            alpha = val
        if best == 1:
            cache[state_id] = best
            return 1
    cache[state_id] = best
    return best


def Min(state: GameState, alpha, beta, cache):
    if state.is_board_flip:
        return -1
    if state.is_no_weights():
        return 0
    state_id = state.to_state()
    if state_id in cache:
        return cache[state_id]
    best = float('inf')
    for pos, weight in state.get_legal_success_move():
        new_state = state.take(pos, weight)
        val = Max(new_state, alpha, beta, cache)
        if val < best:
            best = val
        if val <= alpha:
            cache[state_id] = best
            return best
        if val < beta:
            beta = val
        if best == -1:
            cache[state_id] = best
            return best
    cache[state_id] = best
    return best


if __name__ == '__main__':
    # game = NoTippingGame.INIT_State(4, 10)
    # print(game.get_legal_idxs())
    # print(game)
    #
    # state = GameState.INIT_State(4, 10)
    # state = state.take(-2, 4)
    # state = state.take(0, 3)
    # state = state.take(-3, 1)
    # state = state.take(-1, 1)
    # state = state.take(-5, 3)
    # state = state.take(1, 4)
    # print(state)
    # state = state.take(2, 2)
    # print(state)
    # state = state.take(4, 2)
    # print(state.game.board_state)
    # print(state.is_terminal)
    # print(state.game.get_legal_idxs())
    # print(state)
    #
    # state = state.remove(4)
    # print(state.game.board_state)
    # print(state)
    #
    # state = state.remove(2)
    # print(state.game.board_state)
    # print(state)

    state = GameState.INIT_State(4, 30)
    cache = {}
    start = time.time()
    print(Max(state, float('-inf'), float('inf'), cache))
    print(time.time() - start)
    print(1)

    state_iter = state
    while not state_iter.is_terminal():
        if state_iter.to_play == Player.BLACK:
            print("Player1 state: {}".format(state_iter.to_state()))
            val = Max(state_iter, float('-inf'), float('inf'), cache)
            if val <= -1:
                move_tuples = state_iter.get_legal_success_move()
                if len(move_tuples) == 0:
                    move_tuples = state_iter.get_legal_move_tuple()
                print(len(move_tuples))
                pos, weight = random.choice(move_tuples)
                state_iter = state_iter.take(pos, weight)
                print("Player1: {} {}".format(pos, weight))
                continue
            target_val = val
            for pos, weight in state_iter.get_legal_move_tuple():
                new_state = state_iter.take(pos, weight)
                if Min(new_state, float('-inf'), float('inf'), cache) == target_val:
                    state_iter = new_state
                    print("Player1: {} {}".format(pos, weight))
                    break
        else:
            val = Min(state_iter, float('-inf'), float('inf'), cache)
            print("Player2 state: {}".format(state_iter.to_state()))
            if val >= 1:
                move_tuples = state_iter.get_legal_success_move()
                if len(move_tuples) == 0:
                    move_tuples = state_iter.get_legal_move_tuple()
                pos, weight = random.choice(move_tuples)
                state_iter = state_iter.take(pos, weight)
                print("Player2: {} {}".format(pos, weight))
                continue
            target_val = val
            # print(len(state_iter.get_legal_success_move()))
            for pos, weight in state_iter.get_legal_move_tuple():
                new_state = state_iter.take(pos, weight)
                if Max(new_state, float('-inf'), float('inf'), cache) == target_val:
                    state_iter = new_state
                    print("Player2: {} {}".format(pos, weight))
                    break
    print(state_iter.winner())