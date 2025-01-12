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
def compute_aep(x):
    global i
    global x_total
    global end_time
    global time_running
    x = np.array(x)
    aep = floris_calc(df_wr,x)
    aep = aep*(0.99**constrain1(x))
    f = open('case3/direct/DE/rand4/records'+str(i+1)+'.txt','w')
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
    return -aep

def constrain(x):
    condition = constrain_func(x).constrain()
    return condition[0]

def constrain1(x):
    violation_num = constrain_func1(x).constrain()
    return violation_num

def callback_func(xk,convergence):
    print ('current best layout: {xk}'.format(xk = discrete_objective.decode(xk)))
    

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
init = constrained_sampling_DC(design_space,50,11974,constrain)
discrete_objective = Objective(
        compute_aep,
        variables=[
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
            RandintVar(-16, 19),
            RandintVar(-23, 29),
        ],
    )
bounds = discrete_objective.bounds
optimize_result = differential_evolution(discrete_objective,bounds,strategy='best1bin', init = init, maxiter=10000, popsize=70,disp = True,callback=callback_func,mutation = (0.5,1.2),recombination = 0.7,updating = 'deferred',tol = 1e-10)
decoded_solution = discrete_objective.decode(optimize_result.x)
decoded_fun = discrete_objective(decoded_solution)
end_time = time.time()
time_running = end_time-start_time

np.savetxt('case3/direct/DE/rand10/results.txt',x_total,delimiter = ',',fmt='%.3f')   
f = open('case3/direct/DE/rand10/summary.txt','w')
f.write('Consuming time is '+str(time_running)+' s\n')
f.write('Optimized layout is '+str(solution)+'\n')
f.write('aep of optimized layout is '+str(solution_fitness)+'\n')
f.close()
