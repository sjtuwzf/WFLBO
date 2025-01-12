import os
import numpy as np
import time
import random
import copy

from smt.sampling_methods import LHS
from constrained_sampling.constrained_sampling import *
from case3.constrain import constrain_func
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
from smt.utils.design_space import DesignSpace,IntegerVariable
from acqusition.mes import MaxValueEntropySearch as mes
from optimization.optimization import optimal_search
from floris_calc.floris_calc import *
from acqusition.mes.MaxValueEntropySearch import *
from scipy.optimize import differential_evolution, dual_annealing,NonlinearConstraint

from wrapdisc import Objective
from wrapdisc.var import RandintVar
import multiprocessing
import json

def fit_model(train_X,train_Y,theta0):
    sm = KRG(theta_bounds = [1e-4,1e4],corr = 'pow_exp',print_prediction = False)
    sm.set_training_values(train_X, train_Y)
    sm.train()
    return sm

def objective_function(x):
    x = np.array(x).reshape(-1,50)
    aep_krg = -surrogate_model.predict_values(x)
    if constrain(x[0]) is False:
        aep_krg = aep_krg*0.6
    else:
        aep_krg = aep_krg
    return aep_krg

def acq(x):
    x = np.array(x).reshape(-1,50)
    return -MES.mes(x)

def parallel_compute_acq(num0):
    acq_ = -acq(init_all[num0])
    return acq_
    
def parallel_mutate(mutate_num):
    individual_for_mutate = pop_for_mutate[random.randint(0,19)]
    x = arr_mutate(individual_for_mutate,mutate_num)
    return x
    

def arr_mutate(x,mutate_num):
    x_original = x
    for q in range(mutate_num):
        x_new = copy.copy(x_original)
        mutate_idx = random.randint(0,24)
        x_new[2*mutate_idx] = random.randint(-16,19)
        x_new[2*mutate_idx+1] = random.randint(-23,29)
        while constrain(x_new) is False: 
            x_new = copy.copy(x_original)
            x_new[2*mutate_idx] = random.randint(-16,19)
            x_new[2*mutate_idx+1] = random.randint(-23,29)
    return x_new
    
def parallel_compute_aep(num0):
    aep = floris_calc(df_wr,train_X[num0,:])
    return aep

def constrain(x):
    condition = constrain_func(x).constrain()
    return condition[0]
    

start_time = time.time()
df_wr = pd.read_csv("case3/windrose.csv")

optim_sum = np.zeros((10,52))
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
random_ini = random.sample(range(0,1000),10)
for k in range(10):    
    path = 'case3/hybrid/0.1var/rand_ini_'+str(random_ini[k])
    os.makedirs(path)
for k in range(10):
    num = 250
    dim = 50
    train_X = constrained_sampling_DC(design_space,num,random_ini[k],constrain)
    train_X = train_X.reshape(-1,50)
    with multiprocessing.Pool(processes=60) as pool:
        train_Y = np.array(pool.map(parallel_compute_aep,range(num)))

    initial_set = np.zeros((num,51))
    for i in range(num):
        for j in range(50):
            initial_set [i][j] = train_X[i][j]
        initial_set [i][50] = train_Y[i]
    
    np.savetxt('case3/hybrid/0.1var/rand_ini_'+str(random_ini[k])+'/initial.txt',initial_set,delimiter = ',',fmt='%.3f')    

    optimal_theta = [0.01]
    acq0 = 0.1
    iteration_times = 0
    for i in range(1000):
        f = open('case3/hybrid/0.1var/rand_ini_'+str(random_ini[k])+'/records'+str(i+1)+'.txt','w')
        iteration_times = iteration_times+1
        sbo_seeds = random.sample(range(1,1000),2)
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
        population_x = population_x.reshape(-1,50)
        f.write('population is '+str(population_x)+'\n')
        f.write('aep is '+str(population_y)+'\n')
        f.write('variance of population is '+str(population_var)+'\n')
        init = []
        pop_for_mutate = []
        for j in range(20):
            pop_for_mutate = np.append(pop_for_mutate,population_x[j])
        pop_for_mutate = pop_for_mutate.reshape(-1,50)
        mutate_num = [6]*300
        mutate_num = tuple(mutate_num)
        with multiprocessing.Pool(processes = 60) as pool:
            init = np.array(pool.map(parallel_mutate,mutate_num))
        init = init.reshape(-1,50)
        mutate_num = [3]*100
        mutate_num = tuple(mutate_num)
        with multiprocessing.Pool(processes = 60) as pool:
            population_ = np.array(pool.map(parallel_mutate,mutate_num))
        population_ = population_.reshape(-1,50)
        surrogate_model = fit_model(train_X=train_X, train_Y=train_Y, theta0 = optimal_theta)
        init_predicted = surrogate_model.predict_values(np.array(init))
        init_predicted = np.array(init_predicted).flatten()
        init_idx = np.argsort(init_predicted)[::-1]
        init_opt = []
        for j in range(100):
            init_opt = np.append(init_opt,init[init_idx[j]])
        init_opt = init_opt.reshape(-1,50)
        optimal_theta = surrogate_model.optimal_theta
        discrete_objective = Objective(
            objective_function,
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
        optimal_x = []
        optimal_y = []
        optimized_population=[]
        for t in range(len(sbo_seeds)):
            init_idx = random.sample(range(init_opt.shape[0]),50)
            init_for_opt = []
            for m in range(50):
                init_for_opt = np.append(init_for_opt,init_opt[init_idx[m]])
            init_idx = random.sample(range(init_opt.shape[0]),20)
            for m in range(20):
                init_for_opt = np.append(init_for_opt,population_[init_idx[m]])
            init_for_opt = init_for_opt.reshape(-1,50)
            optimize_result = differential_evolution(discrete_objective,bounds,strategy='best1bin', maxiter=10000, popsize=70,init = init_for_opt,workers = 32,updating = 'deferred',mutation = (0.5,1.2),tol = 1e-3)
        
            decoded_solution = discrete_objective.decode(optimize_result.x)
            decoded_fun = discrete_objective(decoded_solution)
            optimal_x = np.append(optimal_x,decoded_solution)
            optimal_y = np.append(optimal_y,decoded_fun)
            decoded_population = []
            for m in range(70):
                decoded_population = np.append(decoded_population,discrete_objective.decode(optimize_result.population[m]))
            decoded_population = decoded_population.reshape(-1,50)
            optimized_population=np.append(optimized_population,decoded_population)
        optimized_population = optimized_population.reshape(-1,50)
        optimal_x = optimal_x.reshape(-1,50)
      
        optimal_y = -optimal_y
        optimize_population_y = []

        optimize_population_y = surrogate_model.predict_values(np.array(optimized_population))

        optimize_population_y = sorted(optimize_population_y,reverse = True)
        for_mes = []
        for j in range(50):
            for_mes = np.append(for_mes,optimize_population_y[j])
        optimal_check = []
        optimal_check = np.append(train_Y,optimal_y)
        optimal_input = np.zeros(len(sbo_seeds))
        indices = np.argsort(optimal_check)[-len(sbo_seeds):]
        for j in range(len(sbo_seeds)):
            optimal_input[j] = optimal_check[indices[j]]
        
        
        arr_str = json.dumps(optimal_x.tolist())
        f.write('SBO samples are '+'\n'+arr_str+'\n'+'\n')
        f.write('Predicted AEPs of SBO sample are '+'\n'+str(optimal_y)+'\n'+'\n')
        f.write(str(constrain(optimal_x[0]))+' '+str(constrain(optimal_x[1]))+' '+'\n'+'\n')
        optimal_y_index = np.argsort(optimal_y)[::-1]
        
        for j in range(len(optimal_y_index)):
            conditions1 = 'False'
            var_ = surrogate_model.predict_variances(np.array(optimal_x[optimal_y_index[j]]).reshape(-1,50))
            if var_ < 1e-10:
                conditions1 = 'True'
            if conditions1 == 'False':
                optimal_y_aep = floris_calc(df_wr,optimal_x[optimal_y_index[j]])
                train_X = np.append(train_X,[optimal_x[optimal_y_index[j]]],axis = 0)
                train_Y = np.append(train_Y,optimal_y_aep)
                f.write('infilled optimal SBO sample is '+str(optimal_x[optimal_y_index[j],:])+'\n'+'\n')
                f.write(str(constrain(optimal_x[optimal_y_index[j],:]))+'\n'+'\n')
                f.write('Predicted AEP of infilled optimal SBO sample is '+str(optimal_y[optimal_y_index[j]])+'\n'+'\n')
                f.write('AEP of optimal SBO sample is '+str(optimal_y_aep)+'\n'+'\n')    
                break
        if conditions1 == 'True':
            f.write('These samples have existed in the trainning set.\n'+'\n')        
            f.write('optimal f for MES is '+str(for_mes)+'\n')

            MES = MaxValueEntropySearch(surrogate_model, design_space, for_mes)
            discrete_mes = Objective(
                acq,
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
            bounds = discrete_mes.bounds
            infill_x = []
            mes_array = []
            for t in range(len(acq_seeds)):
                init_all = []
                mutate_num = [6]*500
                mutate_num = tuple(mutate_num)
                with multiprocessing.Pool(processes = 60) as pool:
                    init_all = np.array(pool.map(parallel_mutate,mutate_num))
                init_all = init_all.reshape(-1,50)
                with multiprocessing.Pool(processes=60) as pool:
                    init_value = np.array(pool.map(parallel_compute_acq,range(500)))
                init_value = np.array(init_value).flatten()
                init_index = np.argsort(init_value)[::-1]
                init_mes = []
                for m in range(70):
                    init_mes = np.append(init_mes,init_all[init_index[m]])
                    print (init_value[init_index[m]])
                init_mes = np.array(init_mes).reshape(-1,50)
                optimize_result = differential_evolution(discrete_mes,bounds,strategy='best2bin', maxiter=10000, popsize=70,init = init_mes,workers = 32,updating = 'deferred',mutation = (0.5,1.2),tol = 1e-15)
                decoded_solution = discrete_mes.decode(optimize_result.x)
                decoded_fun = discrete_mes(decoded_solution)
                infill_x = np.append(infill_x,decoded_solution)
                mes_array = np.append(mes_array,decoded_fun)
            infill_x = infill_x.reshape(-1,50)
            max_mes_index = np.argmin(mes_array)
            acq1 = min(mes_array)
            if -acq1<0.0001:
                break
            infill_x1 = infill_x[max_mes_index]
            infill_x1 = np.array(infill_x1).reshape(-1,50)
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
        f.write('The current optimal sample is '+str(train_X[np.argmax(train_Y)])+'\n'+'\n')
        f.write('AEP of the current optimal sample is '+str(max(train_Y))+'\n'+'\n')
        f.close()
