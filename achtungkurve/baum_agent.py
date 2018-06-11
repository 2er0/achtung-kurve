import abc
import random
import numpy as np
from typing import Optional
from utils import State, ACTIONS, ACTIONSCALC

import pickle
import os.path


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class SaveState:
    board: [[]]
    action: ""
    result: True

    def __init__(self, board, action, result):
        self.board = board
        self.action = action
        self.result = result


class BaumAgent(Agent):

    data = list()

    def __init__(self):
        if not os.path.isfile("baum/data.txt"):
            return
        with open("baum/data.txt", "rb") as fp:  # Unpickling
            self.data = pickle.load(fp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open("baum/data.txt", "wb") as fp:  # Pickling
            pickle.dump(self.data, fp)

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

        direction = board[x, y] + 1
        l_board = np.rot90(board[xn:xp, yn:yp], direction)

        print(str(l_board).replace('0', '-'))
        choice = random.choice(ACTIONS)
        move = ACTIONSCALC[ACTIONS.index(choice)]

        xt = 2 + move[0]
        yt = 2 + move[1]
        new_pos = l_board[xt, yt]

        state = SaveState(l_board, choice, False if new_pos > 0 else True)
        self.data.append(state)

        return {"move": choice}
