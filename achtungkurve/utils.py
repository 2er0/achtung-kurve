# quick State class
class State:
    position: (1, 2)
    alive: True
    board: [[]]
    last_alive: False
    game_over: False

    def __init__(self, **entries):
        self.__dict__.update(entries)


class SaveState:
    board: [[]]
    action: ""
    result: True

    def __init__(self, board, action, result):
        self.board = board
        self.action = action
        self.result = result


# agent actions
ACTIONS = ["left", "forward", "right"]
ACTIONHOT = {
    "left": 0,
    "forward": 1,
    "right": 2
}

# agent actions in numbers
ACTIONSCALC = [(0, -1), (-1, 0), (0, 1)]
