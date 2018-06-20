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
        return {"move": self.agent_logic(*state)}
    
    def simplify_state(self, state):
        (x, y) = state["position"]
        direction = state["board"][x][y]
        front_dir = (0, 1) if direction==BoardSquare.player_north else \
                     (1, 0) if direction==BoardSquare.player_east else \
                     (0,-1) if direction==BoardSquare.player_south else \
                     (-1,0) if direction==BoardSquare.player_west else None
        if front_dir is None:
            raise ValueError("got unexpected heading of player")
            
        left_dir = (-front_dir[1],front_dir[0])
        right_dir = (front_dir[1],-front_dir[0])
        
        front_tile = state["board"][x + front_dir[0]][y + front_dir[1]]
        left_tile = state["board"][x + left_dir[0]][y + left_dir[1]]
        right_tile = state["board"][x + right_dir[0]][y + right_dir[1]]
        
        return (left_tile, front_tile, right_tile)