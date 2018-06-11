import abc
import random
import numpy as np
from typing import Optional
from utils import State, ACTIONS


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class BaumAgent(Agent):
    def next_move(self, state) -> Optional[dict]:
        state = State(**state)

        if state.last_alive:
            print("I won!! :)")

        if not state.alive:
            print("I'm dead :(")
            return None

        board = np.asarray(state.board, dtype=np.int)
        board = np.pad(board, 1, 'constant', constant_values=9)

        x = state.position[0] + 1
        xn = x-2
        if xn < 0:
            xn = 0
        xp = x+3
        if xp > board.shape[0]:
            xp = board.shape[0]

        y = state.position[1] + 1
        yn = y-2
        if yn < 0:
            yn = 0
        yp = y+3
        if yp > board.shape[1]:
            yp = board.shape[1]

        print([x, y])
        l_board = board[xn:xp, yn:yp]
        print(np.rot90(l_board))

        return {"move": random.choice(ACTIONS)}
