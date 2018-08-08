from achtungkurve.knoll_agent.trongame import TronGame
from achtungkurve.knoll_agent.treeagent import TreeAgent
from achtungkurve.gameengine import Direction
from functools import partial
from deap import algorithms, base, creator, tools, gp
import operator
import numpy as np
import asyncio
import random
import dill as pickle
import sys
import time
import traceback

#sample usage: python knoll_agent.py 3 True True 5

expanded_view = sys.argv[2] == 'True'
opponent_positions = sys.argv[3] == 'True'
memory = int(sys.argv[4])

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2): 
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):     
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    return out1 if condition else out2
    
#definition of classes necessary since bool being instance of int breaks the tree's typings
class MyBool:
    pass

class MyInt:
    pass

#first basic inputs:
#pset = gp.PrimitiveSetTyped("main", [bool, bool, bool], Direction)
minimum = [MyBool, MyInt]*7
if expanded_view:
    expanded = [MyBool]*10
else:
    expanded = []
if opponent_positions:
    positions = [MyInt]*6
else:
    positions = []
inputs = minimum + expanded + positions
if memory > 0:
    inputs = inputs+(inputs+[Direction])*memory
    
pset = gp.PrimitiveSetTyped("main", inputs, Direction)
pset.addPrimitive(operator.not_, [MyBool], MyBool)
pset.addPrimitive(operator.eq, [MyBool, MyBool], MyBool)
pset.addPrimitive(operator.and_, [MyBool, MyBool], MyBool)
pset.addPrimitive(operator.or_, [MyBool, MyBool], MyBool)
pset.addPrimitive(operator.xor, [MyBool, MyBool], MyBool)
pset.addPrimitive(operator.neg, [MyInt], MyInt)
pset.addPrimitive(operator.abs, [MyInt], MyInt)
pset.addPrimitive(operator.eq, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.ne, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.lt, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.le, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.gt, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.ge, [MyInt, MyInt], MyBool)
pset.addPrimitive(operator.add, [MyInt, MyInt], MyInt)
pset.addPrimitive(operator.sub, [MyInt, MyInt], MyInt)
pset.addPrimitive(operator.eq, [Direction, Direction], MyBool)
pset.addPrimitive(if_then_else, [MyBool, Direction, Direction], Direction)
pset.addPrimitive(if_then_else, [MyBool, MyInt, MyInt], MyInt)
pset.addPrimitive(if_then_else, [MyBool, MyBool, MyBool], MyBool)
for direction in Direction:
    pset.addTerminal(direction.value, Direction)
pset.addTerminal(True, MyBool)
pset.addTerminal(False, MyBool)
pset.addEphemeralConstant("distance", lambda : random.randint(-10,10), MyInt)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=5, max_=20)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


loop = None
agents = []
coros = []
def evaluate_agent(individual, cons=1):
    if cons > 1:
        try:
            return (individual.won / individual.played,)
        except (AttributeError, ZeroDivisionError):
            return (np.NaN,)
    else:
        agents[0].set_agent_logic(gp.compile(individual, pset))
        loop.run_forever()
        try:
            individual.won = individual.won + agents[0].won
            individual.played = individual.played + 1
        except AttributeError:
            individual.won = int(agents[0].won)
            individual.played = 1
        
        return (agents[0].won + agents[0].tiles_claimed/(agents[0].board_sum + 1e-10) + 0.1*agents[0].tiles_claimed/(agents[0].board_size + 1e-10),)

def invalidation_decorator(func):
    def wrapper(*args, **kargs):
        children = func(*args, **kargs)
        for child in children:
            del child.fitness.values
        return children
    return wrapper

def selSelfPlay(individuals, k):
    """Play k matches where cons individual play against each other,
    select winners of matches"""
    chosen = []
    for i in range(k):
        aspirants = tools.selection.selRandom(individuals, len(agents))
        for agent, aspirant in zip(agents,aspirants):
            logic = gp.compile(aspirant, pset)
            agent.set_agent_logic(logic)
        for coro in coros:
            loop.run_forever()
        for agent, aspirant in zip(agents, aspirants):
            try:
                aspirant.won = aspirant.won + agent.won
                aspirant.played = aspirant.played + 1
            except AttributeError:
                aspirant.won = int(agent.won)
                aspirant.played = 1
        winner = max(zip(agents,aspirants), key=lambda a: a[0].won)
        if not winner[0].won:
            winners = sorted(zip(agents,aspirants), key=lambda a: a[0].tiles_claimed, reverse=True)
            if winners[0][0].tiles_claimed > winners[1][0].tiles_claimed:
                winner = winners[0]
            else:
                winners = [winner for winner in winners if winner[0].tiles_claimed >= winners[0][0].tiles_claimed]
                winner = min(winners, key=lambda a: a[1].height)
        chosen.append(winner[1])
    return chosen

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=1, max_=3)
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), 50))

if __name__ == "__main__":
    cons = int(sys.argv[1])
    
    gens = 10
    repetitions = 10
    popsize = 100
    pop = toolbox.population(n=popsize+cons-popsize%cons)
    uniform_ratios = (0.5, 0.5)
    insert_ratios = (0.5, 0.5)
    replacement_ratios = (0.5, 0.5)
    optimize_ratios = (0.5, 0.5)
    shrink_ratios = (0.5, 0.5)
    
    def adapt_and_train():
        if len(agents) > 1:
            toolbox.register("select", selSelfPlay)
        else:
            toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.decorate("select", invalidation_decorator)
        toolbox.register("evaluate", evaluate_agent, cons=len(agents))
        
        for ind in pop:
            del ind.fitness.values
        for i in range(repetitions):
            print(i)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 25))
            algorithms.eaSimple(pop, toolbox, *uniform_ratios, gens, stats, verbose=True)
            toolbox.register("mutate", gp.mutInsert, pset=pset)
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 25))
            algorithms.eaSimple(pop, toolbox, *insert_ratios, gens, stats, verbose=True)
            toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 25))
            algorithms.eaSimple(pop, toolbox, *replacement_ratios, gens, stats, verbose=True)
            toolbox.register("mutate", gp.mutEphemeral, mode='one')
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 25))
            algorithms.eaSimple(pop, toolbox, *optimize_ratios, gens, stats, verbose=True)
            toolbox.register("mutate", gp.mutShrink)
            toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), 25))
            algorithms.eaSimple(pop, toolbox, *shrink_ratios, gens, stats, verbose=True)
    
    def get_stats(ind):
        stats = (ind.fitness.values, ind.height)
        return stats
    
    stats = tools.Statistics(get_stats)
    stats.register("avg_f", lambda p: np.nanmean([s[0] for s in p]))
    stats.register("std_f", lambda p: np.nanstd([s[0] for s in p]))
    stats.register("min_f", lambda p: np.nanmin([s[0] for s in p]))
    stats.register("max_f", lambda p: np.nanmax([s[0] for s in p]))
    stats.register("avg_h", lambda p: np.mean([s[1] for s in p]))
    stats.register("min_h", lambda p: np.min([s[1] for s in p]))
    stats.register("max_h", lambda p: np.max([s[1] for s in p]))
    
    #agent plays alone
    #agents = [TreeAgent(expanded_view, opponent_positions, memory)]
    #loop = asyncio.new_event_loop()
    #coros = [loop.create_connection(lambda a=agent: TronGame(a, loop),'localhost', '15554') for agent in agents]
    #t_n_p = []
    #for coro in coros:
    #    t_n_p.append(loop.run_until_complete(coro))
    #adapt_and_train()
    #for transport, protocol in t_n_p:
    #    transport.write_eof()
    #loop.close()
    
    #agent vs. RandomAvoidsWall
    #agents = [TreeAgent(expanded_view, opponent_positions, memory)]
    #loop = asyncio.new_event_loop()
    #coros = [loop.create_connection(lambda a=agent: TronGame(a, loop),'localhost', '15555') for agent in agents]
    #t_n_p = []
    #for coro in coros:
    #    t_n_p.append(loop.run_until_complete(coro))
    #adapt_and_train()
    #for transport, protocol in t_n_p:
    #    transport.write_eof()
    #loop.close()
        
    different individuals against each other
    agents = [TreeAgent(expanded_view, opponent_positions, memory) for i in range(cons)]
    loop = asyncio.new_event_loop()
    coros = [loop.create_connection(lambda a=agent: TronGame(a, loop),'localhost', '15556') for agent in agents]
    t_n_p = []
    for coro in coros:
        t_n_p.append(loop.run_until_complete(coro))
    adapt_and_train()
    for transport, protocol in t_n_p:
        transport.write_eof()
    loop.close()
    
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
    agents[0].agent_logic = gp.compile(sorted_pop[0], pset)
    with open(f'best_individual_{expanded_view}_{opponent_positions}_{memory}.pkl', 'wb') as out_:
        pickle.dump(agents[0], out_, pickle.HIGHEST_PROTOCOL)
        
    with open(f'best_individual_{expanded_view}_{opponent_positions}_{memory}_online.pkl', 'wb') as out_:
        pickle.dump(agents[0], out_, pickle.HIGHEST_PROTOCOL)
        
    print('all pickled up and ready to go!')
    input('press key to start online learning')
    
    while True:
        try:
            repetitions = 1
            #agent vs other GP agents, RL agents, etc.
            agents = [TreeAgent(expanded_view, opponent_positions, memory)]
            loop = asyncio.new_event_loop()
            coros = [loop.create_connection(lambda a=agent: TronGame(agent, loop, print_state=True),'localhost', '15555') for agent in agents]
            t_n_p = []
            for coro in coros:
                t_n_p.append(loop.run_until_complete(coro))
                
            adapt_and_train()
            
            for transport, protocol in t_n_p:
                transport.write_eof()
            loop.close()
            
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
            agents[0].agent_logic = gp.compile(sorted_pop[0], pset)
            with open(f'best_individual_{expanded_view}_{opponent_positions}_{memory}_online.pkl', 'wb') as out_:
                pickle.dump(agents[0], out_, pickle.HIGHEST_PROTOCOL)
                
            print('all pickled up and ready to go!')
            time.sleep(5)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()