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

        if state["last_alive"]:
            print("I won!! :)")

        if not state["alive"]:
            print("I'm dead :(")

        if state["game_over"]:
            print("game_over")

        heading = board[tuple(state["position"])]

        if not self.wall_ahead(board, state["position"], heading):
            return {"move": "forward"}
        elif not self.wall_left(board, state["position"], heading):
            return {"move": "left"}
        else:
            return {"move": "right"}

    def wall_ahead(self, board, position, heading) -> bool:
        move = self.heading_move[heading]
        new_pos = tuple(sum(a) for a in zip(position, move))
        return board[new_pos] != 0

    def wall_left(self, board, position, heading) -> bool:
        heading = ((heading - 2) % 4) + 1
        move = self.heading_move[heading]
        new_pos = tuple(sum(a) for a in zip(position, move))
        return board[new_pos] != 0
