import abc
import numpy as np
import random
from typing import Optional


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
    def next_move(self, state):
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
