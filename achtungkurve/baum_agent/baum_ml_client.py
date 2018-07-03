import asyncio

from baum_agent.baum_ml_agent import BaumMlAgent
from achtungkurve.server import SERVER_PORT
from achtungkurve.client import AgentProtocol


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    with BaumMlAgent() as agent:

        coro = loop.create_connection(lambda: AgentProtocol(agent, loop),
                                      'astra.dbaumi.at', 44480)
        loop.run_until_complete(coro)
        loop.run_forever()
        loop.close()
