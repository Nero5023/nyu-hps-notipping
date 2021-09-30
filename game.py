from enum import Enum
import numpy as np
import math
import random


G_CACHE = {}


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


class NoTippingGame:
    @staticmethod
    def INIT_State(board_len, max_weight_block):
        game = NoTippingGame(board_len, max_weight_block)
        game.board_state = [None] * (board_len + 1)
        game.put_on_index(-4, 3)
        return game

    def __init__(self, board_len, max_weight_block):
        self.board_len = board_len
        self.max_weight_block = max_weight_block
        # TODO: refine use int
        self.board_state = None
        self.left_stand = -3
        self.right_stand = -1
        self.board_weight = 3

    def put_on_index(self, idx, val):
        real_idx = self.to_array_idx(idx)
        self.board_state[real_idx] = val

    def to_array_idx(self, idx):
        mid = self.board_len // 2
        a_idx = mid + idx
        return a_idx

    def to_board_idx(self, idx):
        mid = self.board_len // 2
        return idx - mid

    def left_torque(self):
        if self.board_state is None:
            return 0
        stand_idx = self.to_array_idx(self.left_stand)
        mid_idx = self.to_array_idx(0)
        total_t = (stand_idx - mid_idx)*self.board_weight
        for idx, elem in enumerate(self.board_state):
            if elem is None:
                continue
            torque = (stand_idx - idx)*elem
            total_t += torque
        return total_t

    def right_torque(self):
        if self.board_state is None:
            return 0
        stand_idx = self.to_array_idx(self.right_stand)
        mid_idx = self.to_array_idx(0)
        total_t = (stand_idx - mid_idx)*self.board_weight
        for idx, elem in enumerate(self.board_state):
            if elem is None:
                continue
            torque = (stand_idx - idx)*elem
            total_t += torque
        return total_t

    def __str__(self):
        return "left: {} right: {}".format(self.left_torque(), self.right_torque())


if __name__ == '__main__':
    game = NoTippingGame.INIT_State(60, 10)
    game.put_on_index(-1, 10)
    game.put_on_index(-6, 8)
    print(game)