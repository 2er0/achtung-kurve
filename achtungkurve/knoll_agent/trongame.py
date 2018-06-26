import asyncio
import json

from achtungkurve.client import AgentProtocol
from achtungkurve.agent import Agent
        
class TronGame(AgentProtocol):
    def process_packet(self, packet):
        if packet["game_over"]:
            self.loop.stop()
        move = self.agent.next_move(packet)
        return move
