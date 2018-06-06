class TronGame:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = []
        self.state = []
        self.current_move = {}

    def new_player(self, callback):
        player_num = len(self.players)
        self.players.append(callback)
        print(len(self.players))
        if len(self.players) == self.num_players:
            self.start_game()

        return player_num

    def start_game(self):
        for i, cb in enumerate(self.players):
            cb(f"Starting game! You are player {i}.")

    def move(self, player_num, move):
        self.current_move[player_num] = move

        if len(self.current_move.keys()) == len(self.players):
            self.state.append(self.current_move)
            print("moves so far", self.state)
            print()
            print()

            for cb in self.players:
                cb(str(self.state))

            self.current_move = {}
