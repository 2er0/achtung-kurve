"""
Rlagent for CEM-Agent based on peter_agent/rlagent.py
"""
import asyncio
import multiprocessing

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation, MaxPooling2D, Permute
from keras.optimizers import Adam, RMSprop
from rl.agents.cem import CEMAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.core import Processor, Env, Space
from rl.memory import EpisodeParameterMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy

from agent import Agent
from client import AgentProtocol
from server import SERVER_PORT

def create_dqn_model(env, num_last_frames):
    model = Sequential()

    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


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
        self.eps = 0.7

    def next_move(self, state):
        if self.eps < np.random.rand():
            self.state_queue.put(state)
            action = self.action_queue.get()
        else:
            action = self.action_translate[np.random.randint(0,3)]
            
        self.eps = self.eps * 0.99


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
        # board = self._board_from_state()
        board = np.array(self.state["board"])
        print(str(np.rot90(board)).replace('0', '-'))
        print()

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

        while not self.state["alive"]:
            self.state = self.agent.take_action(None)

        self.alive_opponents = self._count_opponents()
        self.playing_opponents = self.alive_opponents

        return self._board_from_state()

    def _calc_reward(self):
        if self.state["last_alive"] and self.playing_opponents > 0:
            return 1

        score = 0

        if not self.state["alive"]:
            score += -1
        else:
            # small bonus for staying alive, magnitude depending on board size
            # capturing 1/num_players of the board yields 1 reward
            #inner_board_len = np.array(self.state["board"]).shape[0] - 2
            #total_board_air = (inner_board_len ** 2) - 4
            #score += (self.playing_opponents + 1) / total_board_air

            player_list = []
            player_scores = []
            active_player_index = 0
            board = np.array(self.state["board"])
            for index, x in np.ndenumerate(board):
                if x is not 9 and x is not 0:
                    if x < 5:
                        active_player_index = len(player_list)
                    player_list.append(index)
                    player_scores.append(0.)

            for index, x in np.ndenumerate(board):
                if x is 0:
                    player_dist = []
                    for player_pos in player_list:
                        player_dist.append(abs(index[0] - player_pos[0]) + abs(index[1] - player_pos[1]))
                    minimum = min(player_list)
                    indices = [i for i, val in enumerate(player_list) if val == minimum]
                    if len(indices) is 1:
                        player_scores[indices[0]] += 0.001 #0.01

            score += player_scores[active_player_index]

        current_alive = self._count_opponents()

        # extra reward when opponents die
        score += (self.alive_opponents - current_alive) * 0.25

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


def run_agent(agent):
    print("started new process")

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    WINDOW_LENGTH = 1

    num_actions = 3
    view_shape = (21, 21)
    input_shape = (WINDOW_LENGTH,) + view_shape

    env = RestrictedViewTronEnv(agent, 10)

    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation("relu"))

    model.add(Dense(num_actions))
    model.add(Activation('softmax'))
    np.random.seed(2363)

    #policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=2.,
    #                              value_min=.1, value_test=.1, nb_steps=1000000 // 10)

    processor = TronProcessor()

    memory = EpisodeParameterMemory(limit=1000000, window_length=WINDOW_LENGTH)

    cem = CEMAgent(model, nb_actions=num_actions, memory=memory,
                   nb_steps_warmup=50000 // 5, train_interval=4)

    #dqn.compile(Adam(lr=.00025), metrics=["mae"])
    cem.compile()

    weights_filename = 'tmp/dqn_test_weights.h5f'
    checkpoint_weights_filename = 'tmp/dqn_test_weights_{step}.h5f'
    log_filename = 'tmp/dqn_test_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000 // 10)]
    callbacks += [FileLogger(log_filename, interval=10000)]

    def train(transfer=False):
        print(cem.get_config())  # todo save to file

        if transfer:
            cem.load_weights(weights_filename)

        cem.fit(env, callbacks=callbacks, nb_steps=1750000 // 10, log_interval=10000)
        cem.save_weights(weights_filename, overwrite=True)
        cem.test(env, nb_episodes=20, visualize=True)

    def opponent():
        cem.load_weights('tmp/dqn_test_weights.h5f')
        cem.test(env, nb_episodes=200000, visualize=False)

    def test():
        cem.load_weights('tmp/dqn_test_weights.h5f')
        cem.test(env, nb_episodes=20, visualize=True)

    # opponent()
    train() # True
    #test()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    agent = QueueAgent()

    process = multiprocessing.Process(target=run_agent, args=(agent,))
    process.start()

    coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                  'localhost', SERVER_PORT)
    loop.run_until_complete(coro)
    loop.run_forever()
    loop.close()
