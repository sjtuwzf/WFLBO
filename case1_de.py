import os
import numpy as np
import pandas as pd
import math
import time
import random
import pygad
from smt.dempling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerdemplingMethod,MixedIntegerContext
from scipy.optimize import NonlinearConstraint
from smt.utils.design_space import DesignSpace,IntegerVariable
from optimization.optimization import optimal_search
from floris_calc.floris_calc import *
from scipy.optimize import differential_evolution
from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json
i = 0
x_total = []
def compute_aep(x):
    global i
    global x_total
    global end_time
    global time_running
    aep = floris_calc(df_wr,x)
    f = open('case1/direct/de/rand10/records'+str(i+1)+'.txt','w')
    i = i+1
    f.write('demple is '+str(x)+'\n')
    f.write('aep is '+str(aep)+'\n')
    x = np.append(x,aep)
    x_total=np.append(x_total,x).reshape(-1,33)
    end_time = time.time()
    time_running = end_time-start_time
    optimal_x_index = np.argmax(x_total[:,-1])
    optimal_ = x_total [optimal_x_index,:]
    f.write('Consuming time is '+str(time_running)+' s\n')
    f.write('Optimized layout is '+str(optimal_)+'\n') 
    return -aep

def callback_func(xk,convergence):
    print ('current best layout: {xk}'.format(xk = discrete_objective.decode(xk)))
    

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
bounds = discrete_objective.bounds
optimize_result = differential_evolution(discrete_objective,bounds,strategy='best1bin', init = init, maxiter=10000, popsize=70,disp = True,callback=callback_func,mutation = (0.5,1.2),recombination = 0.7,updating = 'deferred')
decoded_solution = discrete_objective.decode(optimize_result.x)
decoded_fun = discrete_objective(decoded_solution)
end_time = time.time()
time_running = end_time-start_time

np.devetxt('case1/direct/de/rand10/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case1/direct/de/rand10/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()
