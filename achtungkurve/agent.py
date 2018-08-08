import abc
import pickle

import numpy as np
import random
from typing import Optional

from utils import State, ACTIONS


class Agent(metaclass=abc.ABCMeta):
    ACTIONS = ["left", "forward", "right"]

    @abc.abstractmethod
    def next_move(self, state):
        pass


class RandomAgent(Agent):
    def next_move(self, state) -> Optional[dict]:

        if state["game_over"]:
            print("game_over")

        if state["last_alive"]:
            print("I won!! :)")

        if not state["alive"]:
            print("I'm dead :(")
            return None

        return {"move": random.choice(self.ACTIONS)}


class AvoidsWallsAgent(Agent):
    def __init__(self):
        super().__init__()
        self.heading_move = {
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
            4: (-1, 0),
        }

    def next_move(self, state):
        board = np.array(state["board"])
        position = tuple(np.array(state["position"]))

        if state["last_alive"]:
            print("I won!! :)")

        if not state["alive"]:
            print("I'm dead :(")

        if state["game_over"]:
            print("game_over")

        heading = board[position]

        return self.move_forward_if_possible(board, position, heading)

    def move_forward_if_possible(self, board, position, heading):
        if not self.wall_ahead(board, position, heading, 1):
            return {"move": "forward"}
        elif not self.wall_ahead(board, position, heading, 0):
            return {"move": "left"}
        else:
            return {"move": "right"}

    def wall_ahead(self, board, position, heading, turn) -> bool:
        heading = ((heading - 2 + turn) % 4) + 1
        move = self.heading_move[heading]
        new_pos = tuple(sum(a) for a in zip(position, move))
        return board[new_pos] != 0


class AvoidsWallsRandomlyAgent(AvoidsWallsAgent):

    count = 1

    def next_move(self, state):

        if state['game_over']:
            print(f'round {self.count} ended')
            self.count += 1

            if self.count > 200:
                exit(0)

        board = np.array(state["board"])
        position = tuple(np.array(state["position"]))

        # if state["last_alive"]:
        #     print("I won!! :)")
        #
        # if not state["alive"]:
        #     print("I'm dead :(")
        #
        # if state["game_over"]:
        #     print("game_over")
        #
        # board = np.array(state["board"])
        # print(str(np.rot90(board)).replace('0', '-'))
        # print()

        if not state["alive"]:
            return None

        heading = board[position]
        rand_turn = random.randint(0, 2)

        if not self.wall_ahead(board, position, heading, rand_turn):
            return {"move": self.ACTIONS[rand_turn]}

        return self.move_forward_if_possible(board, position, heading)


class BaumAgent(Agent):

    clf = None
    pad = 2

    def __init__(self):
        agent = 17
        print(f'BaumAgent: #{agent}')
        with open("baum_agent/agent/" + str(agent) + "/agent.pkl", "rb") as fp:
            other = pickle.load(fp)
            self.clf = other.clf
            self.pad = other.pad

    count = 1

    def next_move(self, state) -> Optional[dict]:

        state = State(**state)

        if state.game_over:
            print(f'round {self.count} ended')
            print(f"Game Over ... win's: {state.wins} | losses: {state.losses}")
            self.count += 1

            if self.count > 200:
                exit(0)

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

        pred = self.clf.predict([f_board])

        choice = ACTIONS[pred[0]]

        return {"move": choice}
