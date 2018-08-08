import asyncio
import json
import numpy as np

from achtungkurve.client import AgentProtocol
from achtungkurve.agent import Agent
        
class TronGame(AgentProtocol):
    def __init__(self, agent: Agent, loop, print_state = False):
        super().__init__(agent, loop)
        self.print_state = print_state
        self.played = 0
        self.won = 0
        
    def process_packet(self, packet):
        if packet["last_alive"]:
            self.agent.won = True
        if packet["game_over"]:
            self.loop.stop()
        if self.print_state:
            print(str(np.rot90(packet["board"])).replace('0', '-'))
            print(f"won {packet['wins']} lost {packet['losses']}")
        move = self.agent.next_move(packet)
        return move
