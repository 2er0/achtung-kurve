import abc
import pickle
import time
import os.path

import numpy as np
from typing import Optional
from utils import State, ACTIONS, ACTIONSCALC, SaveState, ACTIONHOT
from sklearn import tree
from sklearn.externals import joblib


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class BaumMlAgent(Agent):
    clf = tree.DecisionTreeClassifier()
    trained = False
    history = list()
    dir_name = ""

    def __init__(self):
        folders = 0
        for _, dir_names, _ in os.walk('baum/agent'):
            folders += len(dir_names)
            break
        self.dir_name = str(folders)

        if not os.path.exists("baum/agent/"+self.dir_name):
            os.makedirs("baum/agent/"+self.dir_name)

        if not os.path.isfile("baum/dt_3.pkl"):
            exit(1)
        self.clf = joblib.load('baum/dt_3.pkl')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.dump(self.clf, 'baum/dt_4.pkl')
        return

    def next_move(self, state) -> Optional[dict]:
        state = State(**state)

        if state.last_alive:
            self.trained = False
            print("I won!! :)")

        if not state.alive:
            print("I'm dead :(")
            if not self.trained:
                print("I'm training ...")
                self.trained = True
                self.__train_tree()
                self.__save_history()

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

        direction = board[x, y]
        l_board = np.rot90(board[xn:xp, yn:yp], direction)
        f_board = l_board.flatten()
        f_board = np.where(f_board > 0, 9, f_board)

        for choice in ACTIONS:
            move = ACTIONSCALC[ACTIONS.index(choice)]

            xt = 2 + move[0]
            yt = 2 + move[1]
            new_pos = l_board[xt, yt]

            state = SaveState(l_board, choice, False if new_pos > 0 else True)
            self.history.append(state)

        pred = self.clf.predict([f_board])
        # prob_pred = self.clf.predict_proba([f_board])

        # if len(np.where(prob_pred > 0.35)) > 1:
        #    pred[0] = random.choice([0, 2])

        choice = ACTIONS[pred[0]]

        return {"move": choice}

    def __train_tree(self):
        train = []
        labels = []
        for cont in self.history:
            if not cont.result:
                continue
            board = np.asarray(cont.board).flatten()
            board = np.where(board > 0, 9, board)
            lab = ACTIONHOT[cont.action]

            train.append(board)
            labels.append(lab)

        self.clf = self.clf.fit(train, labels)

    def __save_history(self):
        t = time.time()
        with open("baum/agent/"+self.dir_name+"/"+str(t)+".txt", "wb") as fp:  # Pickling
            pickle.dump(self.history, fp)
        self.history = list()
