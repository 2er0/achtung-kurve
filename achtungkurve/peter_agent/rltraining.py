import asyncio
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import numpy as np
from keras import Input, layers, Model
from keras.layers import Conv2D, Flatten, Dense, Activation, Permute, BatchNormalization, add, \
    GlobalAveragePooling2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent, Sequential
from rl.callbacks import ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy

from achtungkurve.client import AgentProtocol
from achtungkurve.peter_agent.rlagent import RestrictedViewTronEnv, TronProcessor, QueueAgent
from achtungkurve.server import SERVER_PORT

TRAIN_DIV = 1


def get_rl_agent(agent):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    WINDOW_LENGTH = 1

    num_actions = 3
    view_shape = (21, 21)
    input_shape = (WINDOW_LENGTH,) + view_shape

    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(32, (5, 5), padding="same", strides=(3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())

    model.add(Dense(1028))
    model.add(Activation("relu"))

    model.add(Dense(num_actions, activation="linear"))

    model.summary()

    np.random.seed(2363)

    policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=2.,
                                  value_min=.1, value_test=.1, nb_steps=1000000 // TRAIN_DIV)

    processor = TronProcessor()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    dqn = DQNAgent(model, nb_actions=num_actions, policy=policy, memory=memory, processor=processor,
                   nb_steps_warmup=50000 // TRAIN_DIV, gamma=.9, target_model_update=1e-2,
                   train_interval=4, delta_clip=1., enable_dueling_network=True, dueling_type="avg")

    dqn.compile(Adam(), metrics=["mae"])

    return dqn


def get_newest_version(model_name):
    model_iteration = 0
    model_path = Path(f"tmp/{model_name}-{model_iteration}/")

    while model_path.exists():
        model_iteration += 1
        model_path = Path(f"tmp/{model_name}-{model_iteration}/")

    return model_iteration - 1


def view_distance_from_dqn(dqn):
    model_cfg = dqn.get_config()["model"]["config"]

    try:
        model_input_size = model_cfg[0]["config"]["batch_input_shape"][2]
    except KeyError:
        model_input_size = model_cfg["layers"][0]["config"]["batch_input_shape"][2]

    assert model_input_size % 2 == 1, "Model not suited for restricted view environment"

    return (model_input_size - 1) // 2


def run_agent(queue_agent, model_name, mode="test", version=None):
    print("started new process")

    dqn = get_rl_agent(queue_agent)

    model_iteration = get_newest_version(model_name) if version is None else version

    transfer_weights_filename = Path(f"tmp/{model_name}-{model_iteration}/") / "dqn_test_weights.h5f"

    if mode == "train":
        model_path = Path(f"tmp/{model_name}-{model_iteration + 1}/")
        model_path.mkdir(parents=True)
        with open(model_path / "model_cfg.json", "w+") as fp:
            json.dump(dqn.get_config(), fp)
    else:
        model_path = Path(f"tmp/{model_name}-{model_iteration}/")
        if not model_path.exists():
            raise ValueError("Model does not exist")

    print(f"Using model {model_path}")

    view_distance = view_distance_from_dqn(dqn)
    env = RestrictedViewTronEnv(queue_agent, view_distance)

    weights_filename = str(model_path / 'dqn_test_weights.h5f')
    checkpoint_weights_filename = str(model_path / "dqn_test_weights_{step}.h5f")

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000 // TRAIN_DIV)]

    def train():
        if transfer_weights_filename.exists():
            dqn.load_weights(transfer_weights_filename)
            print(f"Transfer learning from old model loaded from '{transfer_weights_filename}'")

        dqn.fit(env, callbacks=callbacks, nb_steps=2000000 // TRAIN_DIV, log_interval=10000)
        dqn.save_weights(weights_filename, overwrite=True)
        dqn.save_weights("tmp/dqn_test_weights.h5f", overwrite=True)
        # dqn.test(env, nb_episodes=20, visualize=True)

    def opponent():
        while True:
            dqn.load_weights(weights_filename)
            dqn.test(env, nb_episodes=2000000, visualize=False, verbose=0)

    def test(steps=20, visualize=True):
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=steps, visualize=visualize)

    if mode == "train":
        train()
    elif mode == "test":
        test(20, True)
    elif mode == "evaluate":
        test(1000, False)
    elif mode == "opponent":
        opponent()
    else:
        raise ValueError("Invalid mode")

    queue_agent.quit()


def start_client_process(loop, model_name, mode, server_ip="localhost", server_port=SERVER_PORT, version=None):
    agent = QueueAgent()

    process = multiprocessing.Process(target=run_agent, args=(agent, model_name, mode, version))
    process.start()

    coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                  server_ip, server_port)
    loop.run_until_complete(coro)

    return process


def kill_process(process):
    try:
        os.kill(process.pid, signal.SIGTERM)
    except PermissionError:  # raises when process already dead
        pass


def train_cycle(model_name, latest_version=0):
    while True:
        loop = asyncio.new_event_loop()
        print("start version", latest_version)
        process1 = start_client_process(loop, model_name, "opponent", version=max(0, latest_version - 1))
        process2 = start_client_process(loop, model_name, "opponent", version=max(0, latest_version))
        process3 = start_client_process(loop, model_name, "opponent", version=max(0, latest_version))
        time.sleep(2)
        process4 = start_client_process(loop, model_name, "train")

        loop.run_forever()

        kill_process(process1)
        kill_process(process2)
        kill_process(process3)
        kill_process(process4)

        process1.join()
        process2.join()
        process3.join()
        process4.join()

        latest_version += 1
        loop.close()


if __name__ == "__main__":
    train_cycle("rv10-gamma0.9-duel-4p", 0)

    # loop = asyncio.get_event_loop()
    #
    # process = start_client_process(loop, "rv10-gamma0.9-duel-4p", "test")
    #
    # loop.run_forever()
    #
    # kill_process(process)
    # loop.close()
