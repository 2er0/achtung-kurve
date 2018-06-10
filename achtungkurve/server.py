import asyncio
import json
import random

from achtungkurve.gameengine import TronGame, Player

SERVER_PORT = 15555


class GameProtocol(asyncio.Protocol):
    def __init__(self, game: TronGame):
        self.game = game
        self.transport: asyncio.Transport = None
        self.player: Player = None

    def send_data(self, data: str):
        self.transport.write(json.dumps(data).encode("UTF-8"))

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
        asyncio.ensure_future(self.player.receive_message(json.loads(data.decode("UTF-8"))))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    tron = TronGame(num_players=2, board_size=lambda: random.randint(10, 15),
                    timeout=10, polling_rate=0.5, verbose=False)

    server = loop.run_until_complete(loop.create_server(lambda: GameProtocol(tron), 'localhost', SERVER_PORT))

    print('Serving on {}'.format(server.sockets[0].getsockname()))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
