import asyncio
import json

from achtungkurve.client import AgentProtocol
from achtungkurve.agent import Agent
        
class TronGame(AgentProtocol):
    def __init__(self, agent: Agent, loop):
        self.loop = loop
        self.agent = agent
        self.transport = None

    def process_packet(self, packet):
        if packet["game_over"]:
            self.loop.stop()
        return self.agent.next_move(packet)