from achtungkurve.gameengine import BoardSquare
from achtungkurve.agent import Agent
from typing import Optional
from itertools import chain

class TreeAgent(Agent):
    def __init__(self):
        self.tiles_claimed = 0
        self.agent_logic = None
    
    def next_move(self, state) -> Optional[dict]:
        if not state["alive"]:
            return None

        board_size = (len(state["board"])-2)*(len(state["board"][0])-2)
        self.tiles_claimed = self.tiles_claimed + 1/board_size
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
            
        back_dir = (-front_dir[0],-front_dir[1])
        left_dir = (-front_dir[1],front_dir[0])
        right_dir = (front_dir[1],-front_dir[0])
        left_rear_dir = self.add_tuples(left_dir, back_dir)
        left_front_dir = self.add_tuples(left_dir, front_dir)
        right_front_dir = self.add_tuples(right_dir, front_dir)
        right_rear_dir = self.add_tuples(right_dir, back_dir)
        
        directions = [left_rear_dir, left_dir, left_front_dir, front_dir,
                      right_front_dir, right_dir, right_rear_dir]
        
        tiles_and_distances = [self.calculate_distance(state["board"], (x,y), direction) for direction in directions]
        unpacked_tad = [x for x in chain.from_iterable(tiles_and_distances)]
        simplified = [*unpacked_tad]
        return simplified
    
    def calculate_distance(self, board, position, direction):
        distance = 0
        tile = BoardSquare.air
        tile_pos = position
        while tile == BoardSquare.air:
            tile_pos = self.add_tuples(tile_pos, direction)
            tile = board[tile_pos[0]][tile_pos[1]]
            if tile == BoardSquare.air:
                distance = distance + 1
                
        if tile in [BoardSquare.opponent_east, BoardSquare.opponent_north,
                    BoardSquare.opponent_south, BoardSquare.opponent_west]:
            opponent = True
        else:
            opponent = False 
        return opponent, distance
    
    def add_tuples(self, xs,ys):
     return tuple(x + y for x, y in zip(xs, ys))