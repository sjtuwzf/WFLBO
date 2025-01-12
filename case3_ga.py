import os
import numpy as np
import pandas as pd
import math
import time
import random
import pygad
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
from scipy.optimize import NonlinearConstraint
from smt.utils.design_space import DesignSpace,IntegerVariable
from optimization.optimization import optimal_search
from floris_calc.floris_calc import *
from scipy.optimize import dual_annealing, differential_evolution
from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json
from constrained_sampling.constrained_sampling import *

from case3.constrain import constrain_func as constrain_func
from case3.constrain1 import constrain_func as constrain_func1
i = 0
x_total = []
def compute_aep(on_generation,x,x_idx):
    global i
    global x_total
    global end_time
    global time_running
    x = np.array(x)
    aep = floris_calc(df_wr,x)
    aep = aep*(0.99**constrain1(x))
    f = open('case3/direct/ga/rand10/records'+str(i+1)+'.txt','w')
    i = i+1
    f.write('sample is '+str(x)+'\n')
    f.write('aep is '+str(aep)+'\n') 
    f.write('Constraint violation number is '+str(constrain1(x))+'\n') 
    x = np.append(x,aep)
    x_total=np.append(x_total,x).reshape(-1,51)
    end_time = time.time()
    time_running = end_time-start_time
    optimal_x_index = np.argmax(x_total[:,-1])
    optimal_ = x_total [optimal_x_index,:]
    f.write('Consuming time is '+str(time_running)+' s\n')
    f.write('Optimized layout is '+str(optimal_)+'\n') 
    return aep

def constrain(x):
    condition = constrain_func(x).constrain()
    return condition[0]

def constrain1(x):
    violation_num = constrain_func1(x).constrain()
    #False_count = condition.count(False)
    #print (condition)
    #print (False_count)
    return violation_num

def callback_func(xk,convergence):
    print ('current best layout: {xk}'.format(xk = discrete_objective.decode(xk)))

def on_generation(GA_optimize):
    global last_fitness
    population = GA_optimize.population
    unique_population = []
    for individual in population:
        if not any(np.array_equal(individual,unique) for unique in unique_population):
            unique_population.append(individual)
    while len(unique_population) < len(population):
        num_add = int(len(population)-len(unique_population))
        population_add = constrained_sampling_DC(design_space,num_add*20,1974,constrain)
        random_for_sampling=random.sample(range(0,num_add*20),num_add)
        for i in range(num_add):
            unique_population = np.append(unique_population,population_add[random_for_sampling[i]])
    unique_population = np.array(unique_population).reshape(-1,50)
    GA_optimize.population = unique_population
    print ('Generation = {generation}'.format(generation = GA_optimize.generations_completed))
    print ('Fitness = {fitness}'.format(fitness=GA_optimize.best_solution(pop_fitness=GA_optimize.last_generation_fitness)[1]))
    

start_time = time.time()
df_wr = pd.read_csv("case3/windrose.csv")

design_space0 = DesignSpace([
    IntegerVariable(-16,19),
    IntegerVariable(-23,29),
    IntegerVariable(-16,19),
    IntegerVariable(-23,29),
    IntegerVariable(-16,19),
    IntegerVariable(-23,29),
    IntegerVariable(-16,19),
    IntegerVariable(-23,29),
    IntegerVariable(-16,19),
    IntegerVariable(-23,29),
])
design_space = [design_space0,design_space0,design_space0,design_space0,design_space0]
init = constrained_sampling_DC(design_space,50,10974,constrain)

bounds = [[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29],[-16,19],[-23,29]]
GA_optimize = pygad.GA(num_generations = 100000, num_parents_mating = 10, fitness_func = compute_aep, parent_selection_type = 'sss',gene_space = bounds, gene_type = int, initial_population=init,sol_per_pop = 70, on_generation=on_generation, mutation_type = 'random',crossover_type = 'scattered',mutation_probability = 0.05,mutation_percent_genes=10,crossover_probability = 0.90,keep_elitism = 20)

GA_optimize.run()
solution,solution_fitness,solution_idx=GA_optimize.best_solution(GA_optimize.last_generation_fitness)
end_time = time.time()
time_running = end_time-start_time

np.savetxt('case3/direct/ga/rand2/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case3/direct/ga/rand2/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()
