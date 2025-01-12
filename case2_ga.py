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
    if constrain_func(np.array(x)) >9:
        aep = aep*9/constrain_func(np.array(x))
    else:
        aep = aep*1.0
    f = open('case2/direct/ga_pygad/rand2/records'+str(i+1)+'.txt','w')
    i = i+1
    f.write('sample is '+str(x)+'\n')
    f.write('aep is '+str(aep)+'\n')
    f.write('Distance to zero of optimal is  '+str(constrain_func(np.array(x)))+'\n') 
    x = np.append(x,aep)
    x_total=np.append(x_total,x).reshape(-1,33)
    end_time = time.time()
    time_running = end_time-start_time
    optimal_x_index = np.argmax(x_total[:,-1])
    optimal_ = x_total [optimal_x_index,:]
    f.write('Consuming time is '+str(time_running)+' s\n')
    f.write('Optimized layout is '+str(optimal_)+'\n') 
    f.write('Distance to zero of optimal is  '+str(constrain_func(np.array(optimal_)))+'\n') 
    #optimized_layout_constrain = constrain_func(optimal_)
    return aep

def constrain_func(x):
    turbine_num = int(len(x)/2)
    coordination_x = np.zeros(turbine_num)
    coordination_y = np.zeros(turbine_num)
    
    for i in range(0,turbine_num):
        coordination_x[i] = x[2*i]
        coordination_y[i] = x[2*i+1] 
        
    distance_to_zero = []
    for i in range(turbine_num):
        _distance_to_zero = math.sqrt(coordination_x[i]**2+coordination_y[i]**2)
        distance_to_zero = np.append(distance_to_zero,_distance_to_zero)
    distance_to_zero_max = max(distance_to_zero)
    #print (distance_to_zero_max)
    return distance_to_zero_max

def constrain(x):
    _distance_to_zero = constrain_func(x)
    if _distance_to_zero > 9:
        return False
    else:
        return True

def callback_func(xk,convergence):
    print ('current best layout: {xk}'.format(xk = discrete_objective.decode(xk)))
    
#def on_generation(GA_optimize):
#    global last_fitness
#    print ('Generation = {generation}'.format(generation = GA_optimize.generations_completed))
#    print ('Fitness = {fitness}'.format(fitness=GA_optimize.best_solution(pop_fitness=GA_optimize.last_generation_fitness)[1]))
#    f = open('case1/direct_16turbine/ga_pygad/rand10/population'+str(GA_optimize.generations_completed)+'.txt','w')
#    arr_str=json.dumps(GA_optimize.population.tolist())
#    for j in range(GA_optimize.population.shape[0]):
#        f.write(str(GA_optimize.population[j])+'\n')
#    f.close()
#    last_fitness = GA_optimize.best_solution(pop_fitness = GA_optimize.last_generation_fitness)[1]
    
def on_generation(GA_optimize):
    global last_fitness
    population = GA_optimize.population
    unique_population = []
    for individual in population:
        if not any(np.array_equal(individual,unique) for unique in unique_population):
            unique_population.append(individual)
    while len(unique_population) < len(population):
        num_add = int(len(population)-len(unique_population))
        #random_for_sampling=random.randint(1,1000)
        population_add =  constrained_sampling(design_space,num_add*20,None,constrain)
        random_for_sampling=random.sample(range(0,num_add*20),num_add)
        for i in range(num_add):
            unique_population = np.append(unique_population,population_add[random_for_sampling[i]])
    unique_population = np.array(unique_population).reshape(-1,32)
    GA_optimize.population = unique_population
    print ('Generation = {generation}'.format(generation = GA_optimize.generations_completed))
    print ('Fitness = {fitness}'.format(fitness=GA_optimize.best_solution(pop_fitness=GA_optimize.last_generation_fitness)[1]))
#    f = open('case2/direct/ga_pygad/rand1/population'+str(GA_optimize.generations_completed)+'.txt','w')
#    arr_str=json.dumps(GA_optimize.population.tolist())
#    for j in range(GA_optimize.population.shape[0]):
#        f.write(str(GA_optimize.population[j])+'\n')
#    f.close()
    last_fitness = GA_optimize.best_solution(pop_fitness = GA_optimize.last_generation_fitness)[1]
start_time = time.time()
df_wr = pd.read_csv("windrose_case2.csv")

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
#mi_sampling = MixedIntegerSamplingMethod(LHS,design_space,criterion = 'ese', random_state = 1974)
init = constrained_sampling(design_space,50,2974,constrain)

bounds = [[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9],[-9,9]]
discrete_objective = Objective(
        compute_aep,
        variables=[
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
            RandintVar(-9, 9),
        ],
    )

GA_optimize = pygad.GA(num_generations = 100000, num_parents_mating = 10, fitness_func = compute_aep, parent_selection_type = 'sss',gene_space = bounds, gene_type = int, initial_population=init,sol_per_pop = 70, on_generation=on_generation, mutation_type = 'random',crossover_type = 'scattered',mutation_probability = 0.05,mutation_percent_genes=10,crossover_probability = 0.90,keep_elitism = 20)#, parallel_processing = ['process', 60])
#bounds = discrete_objective.bounds
#con = NonlinearConstraint(constrain_func,-1,9.01)
#optimize_result = differential_evolution(discrete_objective,bounds,strategy='randtobest1exp', maxiter=200, popsize=30,seed = 1,disp = True,callback=callback_func,mutation = (0.5,1.2),recombination = 0.7,updating = 'deferred')#,constraints = [con])
#decoded_solution = discrete_objective.decode(optimize_result.x)
#decoded_fun = discrete_objective(decoded_solution)
GA_optimize.run()
solution,solution_fitness,solution_idx=GA_optimize.best_solution(GA_optimize.last_generation_fitness)
end_time = time.time()
time_running = end_time-start_time

np.savetxt('case2/direct/ga_pygad/rand2/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case2/direct/ga_pygad/rand2/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()