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
from constrained_sampling.constrained_sampling import *
from scipy.optimize import differential_evolution
from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json
i = 0
x_total = []
def compute_aep(on_generation,x,x_idx):
    global i
    global x_total
    global end_time
    global time_running
    aep = floris_calc(df_wr,x)
    f = open('case1/direct/ga/rand2/records'+str(i+1)+'.txt','w')
    i = i+1
    f.write('sample is '+str(x)+'\n')
    f.write('aep is '+str(aep)+'\n')
    x = np.append(x,aep)
    x_total=np.append(x_total,x).reshape(-1,33)
    end_time = time.time()
    time_running = end_time-start_time
    optimal_x_index = np.argmax(x_total[:,-1])
    optimal_ = x_total [optimal_x_index,:]
    f.write('Consuming time is '+str(time_running)+' s\n')
    f.write('Optimized layout is '+str(optimal_)+'\n') 
    f.write('Distance to zero of optimal is  '+str(constrain_func(np.array(optimal_)))+'\n') 
    return aep

    
def on_generation(GA_optimize):
    global last_fitness
    print ('Generation = {generation}'.format(generation = GA_optimize.generations_completed))
    print ('Fitness = {fitness}'.format(fitness=GA_optimize.best_solution(pop_fitness=GA_optimize.last_generation_fitness)[1]))
    f = open('case1/direct/ga/rand10/population'+str(GA_optimize.generations_completed)+'.txt','w')
    arr_str=json.dumps(GA_optimize.population.tolist())
    for j in range(GA_optimize.population.shape[0]):
        f.write(str(GA_optimize.population[j])+'\n')
    f.close()
    last_fitness = GA_optimize.best_solution(pop_fitness = GA_optimize.last_generation_fitness)[1]

start_time = time.time()
df_wr = pd.read_csv("windrose_case1.csv")

design_space = DesignSpace([
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
    IntegerVariable(-9,9),
])

mi_sampling = MixedIntegerSamplingMethod(LHS,design_space,criterion = 'ese', random_state = 33424)
init = mi_sampling(50)

bounds = [[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9]]


GA_optimize = pygad.GA(num_generations = 100000, num_parents_mating = 10, fitness_func = compute_aep, parent_selection_type = 'sss',gene_space = bounds, gene_type = int, initial_population=init,sol_per_pop = 70, on_generation=on_generation, mutation_type = 'random',crossover_type = 'scattered',mutation_probability = 0.05,mutation_percent_genes=10,crossover_probability = 0.90,keep_elitism = 20)
GA_optimize.run()
solution,solution_fitness,solution_idx=GA_optimize.best_solution(GA_optimize.last_generation_fitness)
end_time = time.time()
time_running = end_time-start_time

np.savetxt('case1/direct/ga/rand2/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case1/direct/ga/rand2/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()
