import asyncio
import json
import threading
from json import JSONDecodeError

from achtungkurve.agent import Agent, RandomAgent, AvoidsWallsAgent, AvoidsWallsRandomlyAgent, BaumAgent
from achtungkurve.server import SERVER_PORT


class AgentProtocol(asyncio.Protocol):
    def __init__(self, agent: Agent, loop):
        self.agent = agent
        self.loop = loop
        self.transport = None
        self._lock = threading.Lock()
        self.message_buffer = ""

    def send_data(self, data: dict):
        self.transport.write(json.dumps(data).encode("UTF-8"))

    def connection_made(self, transport):
        self.transport = transport

    def process_packet(self, packet):
        return self.agent.next_move(packet)

    def data_received(self, data):
        self._lock.acquire()

        try:
            data = data.decode("UTF-8")

            split_data = data.split("\0")

            # handle all delimted json dicts
            for received_packet in split_data[:-1]:
                if self.message_buffer:
                    received_packet = self.message_buffer + received_packet
                    self.message_buffer = ""

                received = json.loads(received_packet)

                msg = self.process_packet(received)

                if msg == {"move": "quit"}:
                    self.transport.write_eof()
                    self.loop.stop()
                    return

                if received["alive"] and not received["game_over"] and msg:
                    delimited_json_data = json.dumps(msg) + "\0"
                    self.transport.write(delimited_json_data.encode("UTF-8"))

            if split_data[-1] != "":  # last message not delimited, add to buffer
                self.message_buffer += split_data[-1]

        finally:
            self._lock.release()

    def connection_lost(self, exc):
        print('The server closed the connection')
        print('Stop the event loop')
        self.loop.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    agent = AvoidsWallsRandomlyAgent()

    coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                  'localhost', SERVER_PORT)
    loop.run_until_complete(coro)
    loop.run_forever()
    loop.close()
