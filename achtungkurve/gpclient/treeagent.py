from achtungkurve.gameengine import BoardSquare
from achtungkurve.agent import Agent
from typing import Optional

class TreeAgent(Agent):
    def __init__(self):
        self.tiles_claimed = 0
        self.agent_logic = None
    
    def next_move(self, state) -> Optional[dict]:
        if state["last_alive"]:
            print("I won!! :)")

        if not state["alive"]:
            print("I'm dead :(")
            return None

        self.tiles_claimed = self.tiles_claimed + 1
        state = self.simplify_state(state)
        return {"move": self.agent_logic(state)}
    
    def simplify_state(self, state):
        print(state)
        (x, y) = state["position"]
        direction = state["board"][x][y]
        front_tile = (0, 1) if direction==BoardSquare.player_north else \
                     (1, 0) if direction==BoardSquare.player_east else \
                     (0,-1) if direction==BoardSquare.player_south else \
                     (-1,0) if direction==BoardSquare.player_west else None
        if front_tile is None:
            raise ValueError("got unexpected heading of player")
            
        left_tile = (-front_tile[1],front_tile[0])
        right_tile = (front_tile[1],-front_tile[0])
        
        front_tile = state[x + front_tile[0]][y + front_tile[1]]
        left_tile = state[x + left_tile[0]][y + left_tile[1]]
        right_tile = state[x + right_tile[0]][y + right_tile[1]]
        
        return (left_tile, front_tile, right_tile)