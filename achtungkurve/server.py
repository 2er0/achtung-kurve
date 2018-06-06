import asyncio

from achtungkurve.gameengine import TronGame

SERVER_PORT = 15555


class GameProtocol(asyncio.Protocol):
    def __init__(self, game):
        self.game = game
        self.player_num = None

    def connection_made(self, transport):
        def send_message(message: str):
            transport.write(message.encode("UTF-8"))

        self.player_num = self.game.new_player(send_message)
        print("client connected!", transport.get_extra_info("peername"))

    def connection_lost(self, exc):
        print("connection lost with exception", exc)

    def data_received(self, data):
        print(f"player {self.player_num} sent", data.decode("UTF-8"))
        self.game.move(self.player_num, data.decode("UTF-8"))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    game = TronGame(2)
    server = loop.run_until_complete(loop.create_server(lambda: GameProtocol(game), 'localhost', SERVER_PORT))
    print('Serving on {}'.format(server.sockets[0].getsockname()))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
