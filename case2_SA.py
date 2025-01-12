import os
import numpy as np
import pandas as pd
import math
import time
import random
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
from scipy.optimize import NonlinearConstraint
from smt.utils.design_space import DesignSpace,IntegerVariable
from optimization.optimization import optimal_search
from floris_calc.floris_calc import *
from scipy.optimize import dual_annealing
from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json
i = 0
x_total = []
def compute_aep(x):
    aep = floris_calc(df_wr,x)
    if constrain_func(np.array(x)) >9:
        aep = aep*9/constrain_func(np.array(x))
    else:
        aep = aep*1.0
    f = open('case2/direct/SA/rand2/records'+str(i+1)+'.txt','w')
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
    return -aep

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
#init = constrained_sampling(design_space,50,2974,constrain)
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

#con = NonlinearConstraint(constrain_func,-1,9.01)
optimize_result = dual_annealing(discrete_objective,bounds,no_local_search = True,seed = 1001)#,constraints = [con])
decoded_solution = discrete_objective.decode(optimize_result.x)
decoded_fun = discrete_objective(decoded_solution)
end_time = time.time()
time_running = end_time-start_time

np.savetxt('case1/direct/SA/rand1/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case1/direct/SA/rand1/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()
