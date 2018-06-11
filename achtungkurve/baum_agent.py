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
        xp = x+3

        y = state.position[1] + 1
        yn = y-2
        yp = y+3

        print([x, y])
        l_board = board[xn:xp, yn:yp]
        print(str(np.rot90(l_board)).replace('0', '-'))

        return {"move": random.choice(ACTIONS)}
