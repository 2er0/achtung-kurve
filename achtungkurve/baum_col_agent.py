import abc
import random
import time
import numpy as np
from typing import Optional
from utils import State, ACTIONS, ACTIONSCALC, SaveState

import pickle


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class BaumColAgent(Agent):

    data = list()
    id = 0

    def __init__(self):
        self.id = time.time()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        with open("baum/training/"+str(self.id)+".txt", "wb") as fp:  # Pickling
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
        size = board.shape[0]

        if state.position[0] < 0:
            x = abs(state.position[0]) + 1
            x = size - x
        else:
            x = state.position[0] + 1
        xn = x-2
        xp = x+3

        if state.position[1] < 0:
            y = abs(state.position[1]) + 1
            y = size - y
        else:
            y = state.position[1] + 1
        yn = y-2
        yp = y+3

        direction = board[x, y]
        l_board = np.rot90(board[xn:xp, yn:yp], direction)

        print(str(l_board).replace('0', '-'))

        for choice in ACTIONS:
            move = ACTIONSCALC[ACTIONS.index(choice)]

            xt = 2 + move[0]
            yt = 2 + move[1]
            new_pos = l_board[xt, yt]

            state = SaveState(l_board, choice, False if new_pos > 0 else True)
            self.data.append(state)

        choice = random.choice(ACTIONS)
        return {"move": choice}
