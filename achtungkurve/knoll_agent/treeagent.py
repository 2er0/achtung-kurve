from achtungkurve.gameengine import BoardSquare
from achtungkurve.agent import Agent
from typing import Optional
from itertools import chain
import numpy as np
from collections import deque

class TreeAgent(Agent):
    def __init__(self,expanded_view, opponent_positions, memory):
        self.expanded_view = expanded_view
        self.opponent_positions = opponent_positions
        self.memory = memory
        self.mem_state_length = 14+10*expanded_view+6*opponent_positions+1
        self.set_agent_logic(None)
    
    def set_agent_logic(self, agent_logic):
        self.tiles_claimed = 0
        self.board_sum = 0
        self.won = False
        self.agent_logic = agent_logic
        self.history = deque()
        for i in range(self.memory):
            state = [False,0]*7+[False]*10*self.expanded_view+[0]*6*self.opponent_positions+[None]
            self.history.appendleft(state)
    
    def next_move(self, state) -> Optional[dict]:
        if not state["alive"]:
            return None

        board = np.array(state["board"])[1:-1,1:-1]
        self.board_sum = np.sum((board!=0).astype(int))
        self.tiles_claimed = self.tiles_claimed + 1
        
        state = self.simplify_state(state)
        history = list(chain.from_iterable(self.history))
        move = self.agent_logic(*(state+history))
        
        self.history.appendleft(state+[move])
        self.history.pop()
        
        if move is None:
            move = 'forward'
        return {"move": move}
    
    def simplify_state(self, state):
        (x, y) = state["position"]
        player = state["board"][x][y]
        board = np.rot90(state["board"], k=player-1, axes=(0,1))
        board_w = len(board)
        board_h = len(board[0])
        x,y = (x          ,y          ) if player==1 else \
              (board_h-y-1,x          ) if player==2 else \
              (board_w-x-1,board_h-y-1) if player==3 else \
              (y          ,board_w-x-1) if player==4 else (None,None)
        if x is None or y is None:
            raise ValueError("x or y None")
            
            
        front_dir = (0, 1)
        left_dir = (-1,0)
        right_dir = (1,0)
        left_rear_dir = (-1,-1)
        left_front_dir = (-1,1)
        right_front_dir = (1,1)
        right_rear_dir = (1,-1)
        
        directions = [left_rear_dir, left_dir, left_front_dir, front_dir,
                      right_front_dir, right_dir, right_rear_dir]
        
        tiles_and_distances = [self.calculate_distance(board, (x,y), direction) for direction in directions]
        tiles_and_distances = [x for x in chain.from_iterable(tiles_and_distances)]
        
        if self.expanded_view:
            expanded_pos = [(-1,-2),(-2,-1),(-2,1),(-1,2),
                            (1,2),(2,1),(2,-1),(1,-2),
                            (0,-1),(0,-2)]
        else:
            expanded_pos = []
        
        expanded_pos = [self.add_tuples((x,y),exp) for exp in expanded_pos]
        expanded_pos = [(min(max(0,x),board_w-1),min(max(0,y),board_h-1)) for x,y in expanded_pos]
        expanded_view = [board[exp] != BoardSquare.air for exp in expanded_pos]
        
        if self.opponent_positions:
            opponent_markers = [BoardSquare.opponent_east, BoardSquare.opponent_north,
                              BoardSquare.opponent_south, BoardSquare.opponent_west]
            opponent_pos = [(x,y) for x in range(board_w) for y in range(board_h) \
                            if board[x,y] in opponent_markers]
            rel_opponent_pos = [(ox-x,oy-y) for ox,oy in opponent_pos]
            rel_opponent_pos = list(chain.from_iterable(rel_opponent_pos))
            rel_opponent_pos = rel_opponent_pos + [0]*(6-len(rel_opponent_pos))
        else:
            rel_opponent_pos = []
            
        return tiles_and_distances + expanded_view + rel_opponent_pos
    
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
    
    def mul_tuple(self, xs, k):
     return tuple(x*k for x in xs)