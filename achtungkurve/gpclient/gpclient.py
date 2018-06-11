from achtungkurve.gpclient.trongame import TronGame
from achtungkurve.gameengine import Direction
from functools import partial
from deap import algorithms, base, creator, tools, gp
import operator
import numpy as np

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    return out1 if condition else out2
    
game = TronGame()

pset = gp.PrimitiveSetTyped("main", [bool, bool, bool], Direction)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(if_then_else, [bool, Direction, Direction], Direction)
pset.addTerminal('left', Direction)
pset.addTerminal('forward', Direction)
pset.addTerminal('right', Direction)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

game = TronGame()
def evaluate_agent(individual):
    agent_logic = gp.compile(individual, pset)
    return (game.play_game(agent_logic),)

toolbox.register("evaluate", evaluate_agent)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

if __name__ == "__main__":
    pop = toolbox.population(n=10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    game.phase1()
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, verbose=True)
    game.phase2()
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 160, stats, verbose=True)
    game.phase3()
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 640, stats, verbose=True)