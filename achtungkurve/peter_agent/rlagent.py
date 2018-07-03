import multiprocessing
import time

import numpy as np
from rl.core import Processor, Env, Space

from achtungkurve.agent import Agent


class TronProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 2  # (height, width)
        return np.array(observation).astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 9.
        return processed_batch


class QueueAgent(Agent):
    def __init__(self):
        self.state_queue = multiprocessing.Queue()
        self.action_queue = multiprocessing.Queue()
        self.action_translate = {
            None: None,
            0: "left",
            1: "forward",
            2: "right"
        }

    def quit(self):
        self.action_queue.put("quit")

    def next_move(self, state):
        self.state_queue.put(state)
        action = self.action_queue.get()
        return None if not action else {"move": action}

    def take_action(self, action):
        str_action = self.action_translate[action]
        self.action_queue.put(str_action)
        return self.state_queue.get()


class DiscreteSpace(Space):
    """
    Discrete space analogue to openAI gym's Discrete class.
    {0,1,...,n-1}
    Example usage:
    self.observation_space = DiscreteSpace(2)
    """

    def __init__(self, n):
        self.n = n
        self.shape = ()
        self.dtype = np.int64

    def sample(self, seed=None):
        return np.random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and \
                (x.dtype.kind in np.typecodes['AllInteger'] and
                 x.shape == ()):
            as_int = int(x)
        else:
            return False

        return 0 <= as_int < self.n

    def __repr__(self):
        return f"Discrete({self.n})"

    def __eq__(self, other):
        return self.n == other.n


class TronEnv(Env):
    def __init__(self, agent: QueueAgent):
        self.action_space = DiscreteSpace(3)
        self.agent = agent
        self.alive_opponents = None
        self.playing_opponents = None
        self.state = None

    def render(self, mode='human', close=False):
        board = self._board_from_state()
        # board = np.array(self.state["board"])
        print(str(np.rot90(board)).replace('0', '-'))
        print(f"Won {self.state['wins']}, lost {self.state['losses']}")
        print()
        time.sleep(0.8)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} {type(action)}) invalid."
        self.state = self.agent.take_action(action)

        board = self._board_from_state()

        score = self._calc_reward()

        alive = self.state["alive"]
        game_over = self.state["game_over"]
        done = game_over or not alive

        self.alive_opponents = self._count_opponents()

        return board, score, done, {}

    def reset(self):
        if not self.state:
            self.state = self.agent.state_queue.get()

        while not self.state["alive"] or self.state["game_over"]:
            self.state = self.agent.take_action(None)

        self.alive_opponents = self._count_opponents()
        self.playing_opponents = self.alive_opponents

        return self._board_from_state()

    def _calc_reward(self):
        if self.state["last_alive"] and self.playing_opponents > 0:
            score = 2
        else:
            score = 0

        if not self.state["alive"]:
            score += -2
        else:
            # small bonus for staying alive, magnitude depending on board size
            # capturing 1/num_players of the board yields 1 reward
            inner_board_len = np.array(self.state["board"]).shape[0] - 2
            total_board_air = (inner_board_len ** 2) - 4
            score += (self.playing_opponents + 1) / total_board_air

        current_alive = self._count_opponents()

        # extra reward when opponents die
        score += (self.alive_opponents - current_alive) * 1 / self.playing_opponents

        return score

    def _count_opponents(self) -> int:
        board = np.array(self.state["board"])
        return ((board == 5) | (board == 6) | (board == 7) | (board == 8)).sum()

    def _board_from_state(self):
        return np.array(self.state["board"])


class RestrictedViewTronEnv(TronEnv):
    def __init__(self, agent: QueueAgent, view_distance: int = 1):
        super().__init__(agent)
        self.view_distance = view_distance

    def _board_from_state(self):
        board = np.array(self.state["board"])

        board = np.pad(board, self.view_distance, "constant", constant_values=9)
        x, y = self.state["position"]
        x, y = x + self.view_distance, y + self.view_distance  # position moved due to padding

        board = board[
                x - self.view_distance: x + self.view_distance + 1,
                y - self.view_distance: y + self.view_distance + 1]

        mid = self.view_distance

        if board[mid, mid] == 2:
            board = np.rot90(board, 1)
        elif board[mid, mid] == 3:
            board = np.rot90(board, 2)
        elif board[mid, mid] == 4:
            board = np.rot90(board, 3)

        board[mid, mid] = 1

        return board
