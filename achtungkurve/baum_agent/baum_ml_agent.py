import abc
import os.path
import time
import pickle
from typing import Optional

import numpy as np
from sklearn import tree, naive_bayes

from utils import State, ACTIONS, ACTIONSCALC, SaveState, ACTIONHOT


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


def _new_Tree():
    return naive_bayes.BernoulliNB()

    # [T:12..13{w-201|l-41}..15{w-201|l-88}..17{w201|l-58}]
    # [D:14{w-201|l-98}..T:18{w-58|l-201}]

    # return tree.DecisionTreeClassifier(
    #    criterion="entropy", splitter="random", max_leaf_nodes=128, max_depth=8
    # )

    # [0 .. 11, T:11..16{w-88|l-201}]

    # 19
    # Training: Starting round 202
    # Training: Wins: [137, 50]
    # Battle: Starting round 202
    # Battle: Wins: [170, 28]
    # 20 based on 19 (200 rounds)
    # Training: Starting round 802
    # Training: Wins: [356, 382]
    # Battle: Starting round 202
    # Battle: Wins: [55, 125]


class BaumMlAgent(Agent):
    clf = _new_Tree()

    short_mem_board = list()
    short_mem_labels = list()
    short_mem_size = 256
    long_mem_board = list()
    long_mem_labels = list()

    history = list()
    active = 0
    current = 0
    dir_name = ""
    pad = 3

    def __init__(self):
        load = True
        agent = 75
        self.battle = True

        print("Init client ...")
        folders = 0
        for _, dir_names, _ in os.walk('agent'):
            folders += len(dir_names)
            break
        self.dir_name = str(folders)

        if not self.battle and not os.path.exists("agent/" + self.dir_name):
            os.makedirs("agent/" + self.dir_name)

        if not load:
            print("Generate new agent ...")
            # random.seed(42)
            size = (self.pad - 1) * 2 + 1
            board = np.zeros(shape=(size, size))
            board = np.pad(board, 1, 'constant', constant_values=9)
            board[self.pad, self.pad] = 9
            board = board.flatten()

            for target in ACTIONS:
                self.short_mem_board.append(board)
                self.short_mem_labels.append(ACTIONHOT[target])

            print("Train base model")
            self.clf = self.clf.fit(self.short_mem_board, self.short_mem_labels)
        else:
            print("Load agent from FS ...")
            with open("agent/" + str(agent) + "/agent.pkl", "rb") as fp:
                other = pickle.load(fp)
                self.__apply_past(other)

        print("Init done\n=====================\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # joblib.dump(self, 'agent.pkl')
        if self.battle:
            return
        with open("agent/" + self.dir_name + "/agent.pkl", "wb") as fp:  # Pickling
            pickle.dump(self, fp)
        return

    def next_move(self, state) -> Optional[dict]:
        state = State(**state)

        if state.last_alive:
            print("I won!! :)")

        if state.game_over:
            print("\n=============================================")
            print(f"Game Over ... win's: {state.wins} | losses: {state.losses}")
            print(f"I'm dead :( - Version {self.active}\n")
            print("I'm training ...")
            if not self.battle:
                self.__train_tree()
            self.__save_history()
            print("\n=============================================")
            print(f"I'm running on Version {self.active} now")
            print("=============================================\n")
            #time.sleep(1)

        if not state.alive:
            return None

        pad = self.pad
        board = np.asarray(state.board, dtype=np.int)
        board = np.pad(board, pad, 'constant', constant_values=9)

        x = state.position[0] + pad
        xn = x - pad
        xp = x + pad + 1

        y = state.position[1] + pad
        yn = y - pad
        yp = y + pad + 1

        direction = board[x, y]
        l_board = np.rot90(board[xn:xp, yn:yp], direction)
        f_board = l_board.flatten()
        f_board = np.where(f_board > 0, 9, f_board)

        for choice in ACTIONS:
            move = ACTIONSCALC[ACTIONS.index(choice)]

            xt = pad + move[0]
            yt = pad + move[1]
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
        """ L.__train_tree() -> None -- update short & long memory and build new tree and test old and new """
        train_on = 15
        if len(self.history) > train_on:
            f = -train_on
        else:
            f = 0

        for cont in self.history[f:]:
            if not cont.result:
                continue

            lab = ACTIONHOT[cont.action]

            self.short_mem_board.append(cont.board)
            self.short_mem_labels.append(lab)

        if len(self.short_mem_board) > self.short_mem_size:
            boards = self.short_mem_board[:-self.short_mem_size]
            labels = self.short_mem_labels[:-self.short_mem_size]

            self.short_mem_board = self.short_mem_board[-self.short_mem_size:]
            self.short_mem_labels = self.short_mem_labels[-self.short_mem_size:]

            size = int(len(boards) / 4)
            for _ in range(size):
                index = np.random.randint(len(boards), size=1)[0]
                self.long_mem_board.append(boards.pop(index))
                self.long_mem_labels.append(labels.pop(index))

        boards = self.short_mem_board.copy()
        boards.extend(self.long_mem_board)
        labels = self.short_mem_labels.copy()
        labels.extend(self.long_mem_labels)

        clf_new = _new_Tree()
        clf_new = clf_new.fit(boards, labels)
        self.current += 1
        clf_old_count = 0
        clf_new_count = 0

        for test, lab in zip(boards, labels):
            pred_old = self.clf.predict_proba([test])[0]
            pred_new = clf_new.predict_proba([test])[0]

            if pred_old[lab] > 0.34:
                clf_old_count += 1
            if pred_new[lab] > 0.34:
                clf_new_count += 1

        print(f"Version {self.active} -> {clf_old_count} vs {clf_new_count} <- Version {self.current}")
        if clf_new_count > clf_old_count:
            self.active = self.current
            self.clf = clf_new

    def __save_history(self):
        """ L.__save_history() -> None -- no longer needed """
        self.history = list()
        return

    def __apply_past(self, other):
        """ L.__apply_past(other) -> None -- apply status from other state of agent """
        self.long_mem_board = other.long_mem_board
        self.long_mem_labels = other.long_mem_labels

        self.short_mem_board = other.short_mem_board
        self.short_mem_labels = other.short_mem_labels
        # self.short_mem_size = other.short_mem_size

        self.clf = other.clf
        self.current = other.current
        self.active = other.active

        self.pad = other.pad
