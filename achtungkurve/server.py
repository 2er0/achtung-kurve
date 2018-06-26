import asyncio
import json
import random
import sys

from achtungkurve.gameengine import TronGame, Player

SERVER_PORT = 15555


class GameProtocol(asyncio.Protocol):
    def __init__(self, game: TronGame):
        self.game = game
        self.transport: asyncio.Transport = None
        self.player: Player = None

    def send_data(self, data: dict):
        #print("sending", data)
        delimited_json_data = json.dumps(data) + "\0"
        self.transport.write(delimited_json_data.encode("UTF-8"))

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        self.player = Player(self.send_data)
        connected = self.game.register_player(self.player)
        # todo closing transport here crashes the server, how to refuse connection if not connected?
        print("client connected!", transport.get_extra_info("peername"))

    def connection_lost(self, exc):
        print("connection lost with exception", exc)
        self.player.exit()

    def data_received(self, data):
        split_data = data.decode("UTF-8").split("\0")[:-1]

        for received_packet in split_data:
            received_json = json.loads(received_packet)
            self.player.receive_message(received_json)


def start_tron_server(tron_game: TronGame):
    server = loop.run_until_complete(loop.create_server(lambda: GameProtocol(tron_game), 'localhost', SERVER_PORT))

    print('Serving on {} for {} players'.format(server.sockets[0].getsockname(), tron_game.num_players))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()


if __name__ == "__main__":
    players = 1
    if len(sys.argv) > 1:
        players = int(sys.argv[1])
        
    loop = asyncio.get_event_loop()

    tron = TronGame(num_players=players, board_size=lambda: random.randint(5,10)+5*players,
                    timeout=10, polling_rate=4, verbose=True, last_player_ends_game=True)

    start_tron_server(tron)
