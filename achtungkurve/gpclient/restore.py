from achtungkurve.gpclient.trongame import TronGame
from achtungkurve.gpclient.treeagent import TreeAgent
from achtungkurve.gameengine import Direction, BoardSquare
from achtungkurve.server import SERVER_PORT
from functools import partial
from deap import algorithms, base, creator, tools, gp
import operator
import numpy as np
import asyncio
import random
import dill as pickle
import json
import marshal, types


with open('best_individual.pkl', 'rb') as in_:
    best = pickle.load(in_)
print(best)
#    with open('best_individual.json', 'fb') as in_:
#        best = json.loads(s)
#    print(best)

input()

loop = asyncio.new_event_loop()

coro = loop.create_connection(lambda: TronGame(best, loop),
                          'localhost', SERVER_PORT)

try:
    while(True):
        best.tiles_claimed = 0
        loop.run_until_complete(coro)
        loop.run_forever()
        print(best.tiles_claimed)
finally:
    loop.close()