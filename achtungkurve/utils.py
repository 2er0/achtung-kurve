# quick State class
class State:
    position: (1, 2)
    alive: True
    board: [[]]
    last_alive: False
    game_over: False

    def __init__(self, **entries):
        self.__dict__.update(entries)


# agent actions
ACTIONS = ["left", "forward", "right"]

# agent actions in numbers
ACTIONSCALC = [(0, -1), (-1, 0), (0, 1)]
