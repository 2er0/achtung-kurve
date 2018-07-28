from achtungkurve.knoll_agent.trongame import TronGame
from achtungkurve.knoll_agent.treeagent import TreeAgent
from achtungkurve.gameengine import Direction, BoardSquare
from achtungkurve.server import SERVER_PORT
from functools import partial
from deap import algorithms, base, creator, tools, gp
import numpy as np
import asyncio
import dill as pickle
import traceback
import sys

#sample usage: python restore.py best.pkl

with open(sys.argv[1], 'rb') as in_:
    best = pickle.load(in_)
    
loop = asyncio.new_event_loop()

while True:
    try:
        coro = loop.create_connection(lambda: TronGame(best, loop, True),
                                      'localhost', '15555')
        loop.run_until_complete(coro)
        break
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
print("connected")
try:
    while(True):
        best.tiles_claimed = 0
        loop.run_forever()
        print(best.tiles_claimed)
finally:
    loop.close()