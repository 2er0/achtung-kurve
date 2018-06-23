import asyncio
import multiprocessing

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Activation, MaxPooling2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.core import Processor, Env, Space
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from achtungkurve.agent import Agent
from achtungkurve.client import AgentProtocol
from achtungkurve.server import SERVER_PORT


class TronProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 2  # (height, width)
        return np.array(observation).astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


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
        self.state = None

    def render(self, mode='human', close=False):
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

        board = self.state["board"]
        alive = self.state["alive"]
        game_over = self.state["game_over"]
        last_alive = self.state["last_alive"]
        score = 0.1 if alive else -1.  # todo last_alive bonus

        return np.array(board), score, game_over, {}

    def reset(self):
        if not self.state:
            self.state = self.agent.state_queue.get()

        while not self.state["alive"]:
            self.state = self.agent.take_action(None)

        return np.array(self.state["board"])


def run_agent(agent):
    print("started new process")

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    WINDOW_LENGTH = 4

    num_actions = 3
    board_shape = (10, 10)
    input_shape = (WINDOW_LENGTH,) + board_shape

    env = TronEnv(agent)

    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())

    model.add(Dense(512, activation=None))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(num_actions, activation="linear"))

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.0, nb_steps=1000000)

    processor = TronProcessor()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    dqn = DQNAgent(model, nb_actions=num_actions, policy=policy, memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.9, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=["mae"])

    weights_filename = 'tmp/dqn_test_weights.h5f'
    checkpoint_weights_filename = 'tmp/dqn_test_weights_{step}.h5f'
    log_filename = 'tmp/dqn_test_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=10000)]

    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.

    # dqn.load_weights(weights_filename)

    dqn.test(env, nb_episodes=5, visualize=True)


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
