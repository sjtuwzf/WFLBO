import os
import numpy as np
import time
import random
import pandas as pd

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
from smt.utils.design_space import DesignSpace,IntegerVariable
from acqusition.mes import MaxValueEntropySearch as mes
from optimization.optimization import optimal_search
from floris_calc.floris_calc import *
from acqusition.mes.MaxValueEntropySearch import *
from constrained_sampling.constrained_sampling import *
from scipy.optimize import differential_evolution, NonlinearConstraint

from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json

def fit_model(train_X,train_Y,theta0):
    sm = KRG(corr = 'pow_exp',print_prediction = False)
    sm.set_training_values(train_X, train_Y)
    sm.train()
    return sm

def objective_function(x):
    if constrain_func(np.array(x))>9:
        aep_krg = 0
    else:
        aep_krg = -surrogate_model.predict_values(np.array(x).reshape(-1,32))
    return aep_krg

def arr_mutate(x,mutate_num):            
    mutate_idx = random.sample(range(0,32),mutate_num)
    for q in range(mutate_num):
        x[mutate_idx[q]] = random.randint(-9,9)
    while constrain(x) is False:
        for q in range(mutate_num):
            x[mutate_idx[q]] = random.randint(-9,9)
    return x

def acq(x):
    if constrain_func(np.array(x))>9:
        acq_krg = 0
    else:
        acq_krg = -MES.mes(np.array(x).reshape(-1,32))
    return acq_krg

def parallel_compute_aep(num):
    aep = floris_calc(df_wr,train_X[num,:])
    return aep

def parallel_compute_acq(num):
    acq_ = -acq(init0[num])
    return acq_

def constrain_func(x):
    turbine_num = int(len(x)/2)
    coordination_x = np.zeros(turbine_num)
    coordination_y = np.zeros(turbine_num)
    
    for q in range(0,turbine_num):
        coordination_x[q] = x[2*q]
        coordination_y[q] = x[2*q+1] 
        
    distance_to_zero = []
    for q in range(turbine_num):
        _distance_to_zero = math.sqrt(coordination_x[q]**2+coordination_y[q]**2)
        distance_to_zero = np.append(distance_to_zero,_distance_to_zero)
    distance_to_zero_max = max(distance_to_zero)
    return distance_to_zero_max

def constrain(x):
    _distance_to_zero = constrain_func(x)
    if _distance_to_zero > 9:
        return False
    else:
        return True
    

start_time = time.time()
df_wr = pd.read_csv("windrose_case2.csv")

optim_sum = np.zeros((10,34))

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

random_ini = random.sample(range(1,500),10)
for k in range(10):    
    path = 'case2/hybrid/0.1var/rand_ini_'+str(random_ini[k])
    os.makedirs(path)
    
    
for k in range(10):        
    num = 160
    train_X = constrained_sampling(design_space,num,random_ini[k],constrain)
    train_X = train_X.reshape(-1,32)
    
    with multiprocessing.Pool(processes=32) as pool:
        train_Y = np.array(pool.map(parallel_compute_aep,range(train_X.shape[0])))

    initial_set = np.zeros((train_X.shape[0],33))
    for i in range(train_X.shape[0]):
        for j in range(32):
            initial_set [i][j] = train_X[i][j]
        initial_set [i][32] = train_Y[i]
    
    np.savetxt('case2/hybrid/0.1var/rand_ini_'+str(random_ini[k])+'/initial.txt',initial_set,delimiter = ',',fmt='%.3f')    

    optimal_theta = [0.01]
    acq0 = 0.1
    iteration_times = 0
    
    
    #init = constrained_sampling(design_space,3000,None,constrain)
    for i in range(1000):
        f = open('case2/hybrid/0.1var/rand_ini_'+str(random_ini[k])+'/records'+str(i+1)+'.txt','w')
        iteration_times = iteration_times+1
        sbo_seeds = random.sample(range(1,1000),20)
        acq_seeds = random.sample(range(1,1000),1)
        
        population_index = np.argsort(train_Y)[::-1]
        population_x=[]
        population_y=[]
        for j in range(20):
            population_x = np.append(population_x,train_X[population_index[j]], axis = 0)
            population_y = np.append(population_y,train_Y[population_index[j]])
        population_var = np.var(population_y)    
        if population_var < 0.1:
            break
        population_x = population_x.reshape(-1,32)
        f.write('population is '+str(population_x)+'\n')
        f.write('aep is '+str(population_y)+'\n')
        f.write('variance of population is '+str(population_var)+'\n')
        
        surrogate_model = fit_model(train_X=train_X, train_Y=train_Y, theta0 = optimal_theta)
        init = []
        pop_for_mutate = []
        for j in range(20):
            pop_for_mutate = np.append(pop_for_mutate,population_x[j])
        pop_for_mutate = pop_for_mutate.reshape(-1,32)
        for j in range(3000):
            mutate_idx = random.randint(0,19)
            init = np.append(init,arr_mutate(pop_for_mutate[mutate_idx],8))
        init = init.reshape(-1,32)
        init_predicted = surrogate_model.predict_values(np.array(init).reshape(-1,32))
        init_predicted = np.array(init_predicted).flatten()
        init_idx = np.argsort(init_predicted)[::-1]
        init_opt = []
        for m in range(500):
            init_opt = np.append(init_opt,init[init_idx[m]])
        init_opt = init_opt.reshape(-1,32)
        optimal_theta = surrogate_model.optimal_theta
        discrete_objective = Objective(
            objective_function,
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
        nprocesses1 = 32
        optimal_x = []
        optimal_y = []
        population_mutate = []
        for j in range(100):
            mutate_idx = random.randint(0,19)
            mutate_num = random.randint(0,4)
            population_mutate = np.append(population_mutate,arr_mutate(population_x[mutate_idx],mutate_num))
        population_mutate = population_mutate.reshape(-1,32)
        optimal_x, optimal_y, optimize_population_x = optimal_search(discrete_objective,bounds,design_space,'DE','discrete',20, sbo_seeds,nprocesses1,population_mutate,init_opt)
        optimal_y = -optimal_y
        optimize_population_y = []
        optimize_population_y_var = []
        for j in range(20):
            optimize_population_y = np.append(optimize_population_y,surrogate_model.predict_values(np.array(optimize_population_x[j])))
            optimize_population_y_var0 = np.var(optimize_population_y[j])
            optimize_population_y_var=np.append(optimize_population_y_var,optimize_population_y_var0)
        optimize_population_y = optimize_population_y.reshape(-1,70)
        var_index = np.argmin(optimize_population_y_var)
        
        optimal_check = []
        optimal_check = np.append(train_Y,optimal_y)
        optimal_input = np.zeros(20)
        indices = np.argsort(optimal_check)[-20:]
        for j in range(20):
            optimal_input[j] = optimal_check[indices[j]]
        
        arr_str = json.dumps(optimal_x.tolist())
        f.write('SBO samples are '+'\n'+arr_str+'\n'+'\n')
        f.write('Predicted AEPs of SBO sample are '+'\n'+str(optimal_y)+'\n'+'\n')
        optimal_y_index = np.argsort(optimal_y)[::-1]
        print ('SBO has been done')
        for j in range(len(optimal_y_index)):
            conditions1 = 'False'
            for t in range(train_X.shape[0]):
                if (train_X[t]==optimal_x[optimal_y_index[j]]).all():
                    conditions1 = 'True'
                    break
            if conditions1 == 'False':
                optimal_y_aep = floris_calc(df_wr,optimal_x[optimal_y_index[j]])
                train_X = np.append(train_X,[optimal_x[optimal_y_index[j]]],axis = 0)
                train_Y = np.append(train_Y,optimal_y_aep)
                f.write('infilled optimal SBO sample is '+str(optimal_x[optimal_y_index[j],:])+'\n'+'\n')
                f.write('Predicted AEP of infilled optimal SBO sample is '+str(optimal_y[optimal_y_index[j]])+'\n'+'\n')
                f.write('AEP of optimal SBO sample is '+str(optimal_y_aep)+'\n'+'\n')    
                break
        if j == len(optimal_y_index)-1:
            f.write('These samples have existed in the trainning set.\n'+'\n')        
            f.write('optimal f for MES is '+str(optimize_population_y[var_index])+'\n')
            
            MES = MaxValueEntropySearch(surrogate_model, design_space, optimize_population_y[var_index])
            discrete_mes = Objective(
                acq,
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
            #nprocesses2 = 3
            bounds = discrete_mes.bounds
            infill_x = []
            mes_array = []
            for t in range(len(acq_seeds)):
                
                init0 = []
                for m in range(10000):
                    mutate_idx = random.randint(0,19)
                    mutate_num = random.randint(0,8)
                    init0 = np.append(init0,arr_mutate(pop_for_mutate[mutate_idx],mutate_num))
                init0 = init0.reshape(-1,32)
                
                with multiprocessing.Pool(processes=60) as pool:
                    init0_mes = np.array(pool.map(parallel_compute_acq,range(10000)))
                init0_mes = np.array(init0_mes).flatten()
                init0_idx = np.argsort(init0_mes)[::-1]
                init_mes = []
                for m in range(70):
                    init_mes = np.append(init_mes,init0[init0_idx[m]])
                    print (init0_mes[init0_idx[m]])
                init_mes = np.array(init_mes).reshape(-1,32)
                init_mes = np.append(init_mes,population_x,axis = 0)
                optimize_result = differential_evolution(discrete_mes,bounds,strategy='best2bin', maxiter=15000, popsize=init0.shape[0],init = init_mes,mutation = (0.5,1.2),workers = 60,updating = 'deferred',tol=1e-15)
                decoded_solution = discrete_mes.decode(optimize_result.x)
                decoded_fun = discrete_mes(decoded_solution)
                infill_x = np.append(infill_x,decoded_solution)
                mes_array = np.append(mes_array,decoded_fun)
           
            infill_x = infill_x.reshape(-1,32)
            max_mes_index = np.argmin(mes_array)
            acq1 = min(mes_array)
            if -acq1<1e-3:
                break
            conditions2 = 'False'
            for m in range(train_X.shape[0]):
                if (train_X[j] == infill_x[max_mes_index]).all():
                    conditions2 = 'True'
            if conditions2 == 'False':
                infill_x1 = infill_x[max_mes_index]
                infill_x1 = np.array(infill_x1).reshape(-1,32)
                max_mes_aep = floris_calc(df_wr,infill_x[max_mes_index])
                max_mes_value = surrogate_model.predict_values(infill_x1)
                max_mes_variances = surrogate_model.predict_variances(infill_x1)
                train_X = np.append(train_X,[infill_x[max_mes_index]],axis = 0)
                train_Y = np.append(train_Y,max_mes_aep)
                f.write('Max-value entropy search acqusition function is '+str(-acq1)+'\n'+'\n')
                f.write('Infill sample is '+str(infill_x[max_mes_index])+'\n'+'\n')
                f.write('predicted aep of infilled sample is '+str(max_mes_value)+'\n'+'\n')
                f.write('variances of infilled sample is '+str(max_mes_variances)+'\n'+'\n')
                f.write('AEP of infilled sample is '+str(max_mes_aep)+'\n'+'\n')
            else:
                infill_x1 = infill_x[max_mes_index]
                infill_x1 = np.array(infill_x1).reshape(-1,32)
                max_mes_variances = surrogate_model.predict_variances(infill_x1)
                max_mes_aep = floris_calc(df_wr,infill_x[max_mes_index])
                f.write('The infilled sample has already exists in trainning_set'+'\n'+'\n')
                f.write('Max-value entropy search acqusition function is '+str(-acq1)+'\n'+'\n')
                f.write('Infill sample is '+str(infill_x[max_mes_index])+'\n'+'\n')
                f.write('variances of infilled sample is '+str(max_mes_variances)+'\n'+'\n')
                f.write('AEP of infilled sample is '+str(max_mes_aep)+'\n'+'\n')
        f.write('The current optimal sample is '+str(train_X[np.argmax(train_Y)])+'\n'+'\n')
        f.write('AEP of the current optimal sample is '+str(max(train_Y))+'\n'+'\n')
        f.close()

    solutions_index = np.argmax(train_Y)
    solutions = train_X[solutions_index,:]
    solutions = np.append(solutions,train_Y[solutions_index])
    end_time = time.time()
    time_running = end_time-start_time
    
