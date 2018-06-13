import asyncio

from achtungkurve.baum_agent import Agent, BaumAgent
from achtungkurve.server import SERVER_PORT
from achtungkurve.client import AgentProtocol


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    with BaumAgent() as agent:

        coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                      'localhost', SERVER_PORT)
        loop.run_until_complete(coro)
        loop.run_forever()
        loop.close()
