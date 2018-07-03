import asyncio
import time
from enum import Enum, IntEnum
from itertools import permutations
from typing import Union, Callable, List

import numpy as np


# Coordinates start at top left
# y values go from south to north
# x values go from west to east
class BoardSquare(IntEnum):
    air = 0
    player_north = 1
    player_east = 2
    player_south = 3
    player_west = 4
    opponent_north = 5
    opponent_east = 6
    opponent_south = 7
    opponent_west = 8
    wall = 9


class Heading(IntEnum):
    north = 0
    east = 1
    south = 2
    west = 3


class Direction(Enum):
    left = "left"
    forward = "forward"
    right = "right"


class Player:

    def __init__(self, client_callback: Callable[[dict], None]):
        """This class connects game and server protocol. Each instance of
        game protocol will only modify their own player instance. The game
        will poll this player for their *moved* flag to see if that player
        has moved since the last update.
        This class also handles communication back to their agent. It will
        modify the board so that its own position is marked differently than
        his opponents positions before sending it back to the agent.

        :param client_callback: Calling this function with a dictionary
        will send a json encoded version of it to the agent.
        """
        self.client_callback = client_callback
        self.playing = True
        self.moved = False
        self.alive = True

        self.heading: Heading = None
        self.x: int = None
        self.y: int = None
        self.games_won: int = 0
        self.games_lost: int = 0

    def send_data(self, data: dict):
        board = data["board"]
        board[self.x, self.y] = self.heading + 1

        data["position"] = (self.x, self.y)
        data["alive"] = self.alive
        data["board"] = board.tolist()  # np.ndarray is not json serializable
        data["wins"] = self.games_won
        data["losses"] = self.games_lost

        self.client_callback(data)

    def receive_message(self, msg: dict):
        # updates heading and position

        if not self.alive:
            return

        if self.moved:
            raise ValueError(f"Player at position ({self.x},{self.y}) tried to move twice")

        direction = Direction(msg["move"])  # convert to enum

        self._turn(direction)
        self._take_step_forward()
        self.moved = True

    def _turn(self, direction: Direction):
        direction_int = {
            Direction.left: -1,
            Direction.forward: 0,
            Direction.right: 1
        }

        self.heading = (self.heading + direction_int[direction]) % 4

    def _take_step_forward(self):
        heading_move = {
            Heading.north: (0, 1),
            Heading.east: (1, 0),
            Heading.south: (0, -1),
            Heading.west: (-1, 0),
        }

        # hacky way of updating positions
        self.x, self.y = [sum(a) for a in zip((self.x, self.y), heading_move[self.heading])]

    def exit(self):
        self.playing = False


class TronGame:

    def __init__(self, num_players=4, board_size: Union[Callable[[], int], int] = 30,
                 polling_rate: float = 0., timeout: float = 30.,
                 verbose: bool = False, last_player_ends_game: bool = True,
                 last_player_mvp: bool = False):
        """Implementation of the Tron Light Cycles game.

        :param num_players: Number of players that can connect at once. Limited to 1-4
        :param board_size: Either an int specifying one side-length of the board or a
        callable that takes no parameters and returns an int (e.g. lambda:
        random.randint(15, 30))
        :param polling_rate: Controls the speed of the game loop (in Hertz). If set to
        0, it will refresh as quickly as possible.
        :param timeout: How long to wait for each Player to move. If the timeout is
        reached, it will stop the round and begin the next one
        :param verbose: Wether to print the game state after each step
        :param last_player_ends_game: In multiplayer games, controls whether to stop
            the game when only one player is alive (True) or to keep playing until
            all players are dead
        :param last_player_mvp: In multiplayer games, controls whether to stop
            the game when the first player (the MVP) dies. This speeds up training
            for that player.
        """
        if 0 > num_players > 4:
            raise ValueError(f"Only 1-4 players are supported, got {num_players}")

        self.timeout = timeout
        self._step_sleep_time = polling_rate if polling_rate == 0 else 1 / polling_rate
        self.board_size = board_size
        self.num_players = num_players
        self.verbose = verbose
        self.last_player_ends_game = last_player_ends_game
        self.first_player_mvp = last_player_mvp

        self.round = 0
        self.players: List[Player] = []
        self.board = None
        self.game_over = False

    def register_player(self, player: Player):
        if len(self.players) == self.num_players:
            return False

        self.players.append(player)

        if len(self.players) == self.num_players:
            asyncio.ensure_future(self._start_game_loop())

        return True

    async def _start_game_loop(self):
        for player in self.players:
            player.games_won = 0

        while len(self.players) > 0 and all(p.playing for p in self.players):
            self.round += 1

            if self.verbose:
                print(f"Starting round {self.round}")
                print(f"Wins: {[p.games_won for p in self.players]}")
                print()

            self._reset_game()
            self._broadcast_state()

            timeout_start = time.time()

            while not self.game_over and all(p.playing for p in self.players):
                if time.time() - timeout_start > self.timeout:
                    self.game_over = True
                    print("No response from agent, game timed out...")

                if all(p.moved or not p.alive for p in self.players):
                    self._update_board()

                    all_dead = self._num_alive() < 1
                    is_multiplayer = len(self.players) > 1
                    last_man_standing = self._num_alive() <= 1 and self.last_player_ends_game
                    mvp_dead = self.first_player_mvp and not self.players[-1].alive

                    if all_dead or (is_multiplayer and last_man_standing) or mvp_dead:
                        self._add_wins()
                        self.game_over = True

                    if self.verbose:
                        # quick way to get a better board view
                        print(str(np.rot90(self.board)).replace('0', '-'))
                        print()

                    self._broadcast_state()

                    timeout_start = time.time()

                await asyncio.sleep(self._step_sleep_time)

        print("Player disconnected, game ends")
        print(f"Wins: {[p.games_won for p in self.players]}")
        print()

        self._remove_inactive_players()

    def _num_alive(self):
        return sum(p.alive and p.playing for p in self.players)

    def _reset_game(self):
        self.game_over = False

        board_size = self.board_size() if callable(self.board_size) else self.board_size

        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.board[0, :] = BoardSquare.wall
        self.board[-1, :] = BoardSquare.wall
        self.board[:, 0] = BoardSquare.wall
        self.board[:, -1] = BoardSquare.wall

        start_positions = (2, 2), (board_size - 3, board_size - 3), \
                          (2, board_size - 3), (board_size - 3, 2)
        start_headings = [BoardSquare.opponent_north, BoardSquare.opponent_south,
                          BoardSquare.opponent_east, BoardSquare.opponent_west]

        start_headings = np.random.choice(start_headings, size=(len(start_positions),))
        starts = list(zip(start_positions, start_headings))
        starts = np.random.permutation(starts)

        for (position, heading), player in zip(starts, self.players):
            self.board[position] = heading
            player.heading = heading - 5  # todo
            player.x, player.y = position
            player.alive = True
            player.moved = False

    def _update_board(self):
        # previous player position turns to wall
        self.board[(self.board == BoardSquare.opponent_north) |
                   (self.board == BoardSquare.opponent_east) |
                   (self.board == BoardSquare.opponent_south) |
                   (self.board == BoardSquare.opponent_west)] = BoardSquare.wall

        self._check_collisions()

        # maps player heading to BoardSquare.opponent_*
        heading_to_opponent = {Heading.north: BoardSquare.opponent_north,
                               Heading.east: BoardSquare.opponent_east,
                               Heading.south: BoardSquare.opponent_south,
                               Heading.west: BoardSquare.opponent_west}

        for player in self.players:
            if self._check_alive(player):
                self.board[player.x, player.y] = heading_to_opponent[player.heading]

    def _broadcast_state(self):
        for player in self.players:
            player.moved = False

            data = {
                "game_over": self.game_over,
                "board": self.board.copy(),
                "last_alive": self._num_alive() == 1 and player.alive
            }

            player.send_data(data)

    def _check_collisions(self):
        for (player1, player2) in permutations(self.players, 2):
            if player1.x == player2.x and player1.y == player2.y:
                self.board[player1.x, player1.y] = BoardSquare.wall
                player1.alive = False
                player2.alive = False

    def _check_alive(self, player: Player) -> bool:
        if not player.alive:
            return False

        if self.board[player.x, player.y] == BoardSquare.wall:
            player.alive = False

        return player.alive

    def _remove_inactive_players(self):
        self.players = [player for player in self.players if player.playing]

    def _add_wins(self):
        if len(self.players) > 1 and sum(p.alive for p in self.players) == 1:
            for player in self.players:
                if player.alive:
                    player.games_won += 1
                else:
                    player.games_lost += 1

