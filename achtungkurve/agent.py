import abc
import random


class Agent(metaclass=abc.ABCMeta):
    ACTIONS = ["left", "forward", "right"]

    @abc.abstractmethod
    def next_move(self, state):
        pass


class RandomAgent(Agent):
    def next_move(self, state):
        return random.choice(self.ACTIONS)
