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
    hist_board = list()
    hist_labels = list()
    history = list()
    aktive = 0
    current = 0
    dir_name = ""

    def __init__(self):
        print("Init client ...")
        folders = 0
        for _, dir_names, _ in os.walk('baum/agent'):
            folders += len(dir_names)
            break
        self.dir_name = str(folders)

        if not os.path.exists("baum/agent/" + self.dir_name):
            os.makedirs("baum/agent/" + self.dir_name)

        """
        if not os.path.isfile("baum/dt_3.pkl"):
            exit(1)
        print("Load base model ...")
        self.clf = joblib.load('baum/dt_3.pkl')
        """

        print("Load base data for future trainings ...")
        examples = 10
        for filename in os.listdir("baum/training"):
            with open("baum/training/" + filename, "rb") as fp:  # Unpickling
                data = pickle.load(fp)

                collect = [0, 0, 0]

                for cont in data:
                    if not cont.result:
                        continue

                    lab = ACTIONHOT[cont.action]
                    collect[lab] += 1

                    if collect[lab] < examples:
                        board = np.asarray(cont.board).flatten()
                        board = np.where(board > 0, 9, board)

                        self.hist_board.append(board)
                        self.hist_labels.append(lab)

                    if all(i >= examples for i in collect):
                        break

        print("Train base model")
        self.clf = self.clf.fit(self.hist_board, self.hist_labels)

        print("Init done\n=====================\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.dump(self.clf, 'baum/dt_4.pkl')
        return

    def next_move(self, state) -> Optional[dict]:
        state = State(**state)

        if state.last_alive:
            self.trained = False
            # print("I won!! :)")

        if not state.alive:
            print(f"I'm dead :( - Version {self.aktive}")
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

            state = SaveState(f_board, choice, False if new_pos > 0 else True)
            self.history.append(state)

        pred = self.clf.predict([f_board])
        # prob_pred = self.clf.predict_proba([f_board])

        # if len(np.where(prob_pred > 0.35)) > 1:
        #    pred[0] = random.choice([0, 2])

        choice = ACTIONS[pred[0]]

        return {"move": choice}

    def __train_tree(self):
        train_on = 10
        if len(self.history) > train_on:
            f = -train_on
        else:
            f = 0

        for cont in self.history[f:]:
            if not cont.result:
                continue
            # board = np.asarray(cont.board).flatten()
            # board = np.where(board > 0, 9, board)
            lab = ACTIONHOT[cont.action]

            self.hist_board.append(cont.board)
            self.hist_labels.append(lab)

        clf_new = tree.DecisionTreeClassifier()
        clf_new = clf_new.fit(self.hist_board, self.hist_labels)
        self.current += 1
        clf_old_count = 0
        clf_new_count = 0

        for test, lab in zip(self.hist_board, self.hist_labels):
            pred_old = self.clf.predict_proba([test])[0]
            pred_new = clf_new.predict_proba([test])[0]

            if pred_old[lab] > 0.35:
                clf_old_count += 1
            if pred_new[lab] > 0.35:
                clf_new_count += 1

        print(f"{clf_old_count} vs {clf_new_count} - Version {self.current}")
        if clf_new_count >= clf_old_count:
            self.aktive = self.current
            self.clf = clf_new

    def __save_history(self):
        t = time.time()
        with open("baum/agent/" + self.dir_name + "/" + str(t) + ".txt", "wb") as fp:  # Pickling
            pickle.dump(self.history, fp)
        self.history = list()
