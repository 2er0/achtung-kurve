import abc
import random
from typing import Optional


class Agent(metaclass=abc.ABCMeta):
    ACTIONS = ["left", "forward", "right"]

    @abc.abstractmethod
    def next_move(self, state):
        pass


class RandomAgent(Agent):
    def next_move(self, state) -> Optional[dict]:

        if state["last_alive"]:
            print("I won!! :)")

        if not state["alive"]:
            print("I'm dead :(")
            return None

        return {"move": random.choice(self.ACTIONS)}
