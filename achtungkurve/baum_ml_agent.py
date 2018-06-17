import abc
import numpy as np
from typing import Optional
from utils import State, ACTIONS, ACTIONSCALC, SaveState
from sklearn import tree
from sklearn.externals import joblib

import os.path


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class BaumMlAgent(Agent):
    clf = tree.DecisionTreeClassifier()

    def __init__(self):
        if not os.path.isfile("baum/dt_2.pkl"):
            return
        self.clf = joblib.load('baum/dt_2.pkl')
        print(self.clf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

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
        xn = x - 2
        xp = x + 3

        if state.position[1] < 0:
            y = abs(state.position[1]) + 1
            y = size - y
        else:
            y = state.position[1] + 1
        yn = y - 2
        yp = y + 3

        direction = board[x, y] + 1
        l_board = np.rot90(board[xn:xp, yn:yp], direction)
        f_board = l_board.flatten()

        pred = self.clf.predict([f_board])
        tmp = self.clf.predict_proba([f_board])

        choice = ACTIONS[pred[0]]

        return {"move": choice}
