import abc
import random
from typing import Optional
from utils import State, ACTIONS


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_move(self, state):
        pass


class BaumAgent(Agent):
    def next_move(self, state) -> Optional[dict]:
        state = State(**state)

        if state.last_alive:
            print("I won!! :)")

        if not state.alive:
            print("I'm dead :(")
            return None

        return {"move": random.choice(ACTIONS)}
