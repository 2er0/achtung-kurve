import asyncio
import random
import time

from achtungkurve.agent import Agent, RandomAgent
from achtungkurve.server import SERVER_PORT


class AgentProtocol(asyncio.Protocol):
    def __init__(self, agent: Agent, loop):
        self.agent = agent  # todo able to swap agent without restarting server and reconnecting
        self.loop = loop
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        print('Data received: {!r}'.format(data.decode()))
        time.sleep(random.uniform(1, 5))
        msg = self.agent.next_move(data.decode())
        print("sending", msg)
        self.transport.write(msg.encode("UTF-8"))

    def connection_lost(self, exc):
        print('The server closed the connection')
        print('Stop the event loop')
        self.loop.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    agent = RandomAgent()

    coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                  'localhost', SERVER_PORT)
    loop.run_until_complete(coro)
    loop.run_forever()
    loop.close()
