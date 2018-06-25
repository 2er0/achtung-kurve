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

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    return out1 if condition else out2

#pset = gp.PrimitiveSetTyped("main", [bool, bool, bool], Direction)
inputs = [bool, int]*7
pset = gp.PrimitiveSetTyped("main", inputs, Direction)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.eq, [int, int], bool)
pset.addPrimitive(operator.ne, [int, int], bool)
pset.addPrimitive(operator.lt, [int, int], bool)
pset.addPrimitive(operator.le, [int, int], bool)
pset.addPrimitive(operator.gt, [int, int], bool)
pset.addPrimitive(operator.ge, [int, int], bool)
pset.addPrimitive(operator.add, [int, int], int)
pset.addPrimitive(operator.sub, [int, int], int)
pset.addPrimitive(if_then_else, [bool, Direction, Direction], Direction)
pset.addPrimitive(if_then_else, [bool, int, int], int)
pset.addPrimitive(if_then_else, [bool, bool, bool], bool)
for direction in Direction:
    pset.addTerminal(direction.value, Direction)
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

pset.addEphemeralConstant("distance", lambda : random.randint(0,10), int)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=0, max_=0)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


loop = None
agent = TreeAgent()
coro = None
def evaluate_agent(individual):
    agent.tiles_claimed = 0
    agent.agent_logic = gp.compile(individual, pset)
    loop.run_until_complete(coro)
    loop.run_forever()
    return (agent.tiles_claimed,)

def invalidation_decorator(func):
    def wrapper(*args, **kargs):
        children = func(*args, **kargs)
        for child in children:
            del child.fitness.values
        return children
    return wrapper

toolbox.register("evaluate", evaluate_agent)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("select", invalidation_decorator)
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), 50))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 50))

if __name__ == "__main__":
    gens = 20
    pop = toolbox.population(n=100)
    readapt_ratios = (0.2, 0.8)
    optimize_ratios = (0.8, 0.2)
    
    def adapt_and_train():
        for ind in pop:
            del ind.fitness.values
        algorithms.eaSimple(pop, toolbox, *readapt_ratios, gens, stats, verbose=True)
        algorithms.eaSimple(pop, toolbox, *optimize_ratios, gens, stats, verbose=True)
    
    def get_stats(ind):
        stats = (ind.fitness.values, ind.height)
        return stats
    
    stats = tools.Statistics(get_stats)
    stats.register("avg_f", lambda p: np.mean([s[0] for s in p]))
    stats.register("std_f", lambda p: np.std([s[0] for s in p]))
    stats.register("min_f", lambda p: np.min([s[0] for s in p]))
    stats.register("max_f", lambda p: np.max([s[0] for s in p]))
    stats.register("avg_h", lambda p: np.mean([s[1] for s in p]))
    stats.register("min_h", lambda p: np.min([s[1] for s in p]))
    stats.register("max_h", lambda p: np.max([s[1] for s in p]))
    
    loop = asyncio.new_event_loop()
    coro = loop.create_connection(lambda: TronGame(agent, loop),
                              'localhost', SERVER_PORT)
    adapt_and_train()
    loop.close()
    
    input()
    
    loop = asyncio.new_event_loop()
    coro = loop.create_connection(lambda: TronGame(agent, loop),
                              'localhost', SERVER_PORT)
    adapt_and_train()
    loop.close()
    
    input()
    
    loop = asyncio.new_event_loop()
    coro = loop.create_connection(lambda: TronGame(agent, loop),
                              'localhost', SERVER_PORT)
    adapt_and_train()
    loop.close()
    
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
    
    with open('best_individual.pkl', 'wb') as out_:
        pickle.dump(agent, out_, pickle.HIGHEST_PROTOCOL)
        
    print('all pickled up and ready to go!')