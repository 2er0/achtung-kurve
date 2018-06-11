class TronGame(object):
    values = [([False, False, False],['left','forward','right']),
              ([False, False, True],['left','forward']),
              ([False, True, False],['left','right']),
              ([False, True, True],['left']),
              ([True, False, False],['forward','right']),
              ([True, False, True],['forward']),
              ([True, True, False],['right']),
              ([True, True, True],[])]
    
    def play_game(self, agent_logic):
        tiles_claimed = 0.0
        for inputs, allowed in self.values:
            output = agent_logic(*inputs)
            if output not in allowed:
                return tiles_claimed
            tiles_claimed  = tiles_claimed+1
        return tiles_claimed
    
    def phase1(self):
        pass
    
    def phase2(self):
        pass
    
    def phase3(self):
        pass