import asyncio
import json

from achtungkurve.agent import Agent, RandomAgent, AvoidsWallsAgent
from achtungkurve.server import SERVER_PORT


class AgentProtocol(asyncio.Protocol):
    def __init__(self, agent: Agent, loop):
        self.agent = agent  # todo able to swap agent without restarting server and reconnecting
        self.loop = loop
        self.transport = None

    def send_data(self, data: dict):
        self.transport.write(json.dumps(data).encode("UTF-8"))

    def connection_made(self, transport):
        self.transport = transport

    def process_packet(self, packet):
        return self.agent.next_move(packet)

    def data_received(self, data):
        split_data = data.decode().split("\0")[:-1]
        for received_packet in split_data:
            received = json.loads(received_packet)
            msg = self.process_packet(received)

            if msg:
                self.transport.write(json.dumps(msg).encode("UTF-8"))

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
