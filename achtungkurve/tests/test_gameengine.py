import unittest
from achtungkurve.gameengine import TronGame, Player, Heading


class TestTronGame(unittest.TestCase):

    def _dummy_callback(self, _):
        pass

    def test_player_init(self):
        msg = {"msg": "test"}

        def assertion_callback(client_msg):
            nonlocal msg
            self.assertEqual(client_msg, msg)

        player = Player(assertion_callback)

        player.client_callback(msg)

        self.assertTrue(player.playing)
        self.assertTrue(player.alive)
        self.assertFalse(player.moved)

    def test_player_step(self):
        player = Player(self._dummy_callback)
        player.x = 3
        player.y = 3
        player.heading = Heading.north  # towards positive y

        msg = {"move": "forward"}

        player.receive_message(msg)

        self.assertEqual(player.x, 3)
        self.assertEqual(player.y, 4)


