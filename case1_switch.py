import os
import numpy as np
import time
import random

from smt.sampling_methods import LHS
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
    sm = KRG(theta0=theta0,corr = 'pow_exp',print_prediction = False)
    sm.set_training_values(train_X, train_Y)
    sm.train()
    return sm

def objective_function(x):
    return -surrogate_model.predict_values(np.array(x).reshape(-1,32))

def acq(x):
    x = np.array(x).reshape(-1,32)
    return -MES.mes(x)

def parallel_compute_acq(num):
    acq_ = -acq(init_all[num])
    return acq_

def arr_mutate(x):
    mutate_num = random.randint(0,16)
    mutate_idx = random.sample(range(0,32),mutate_num)
    for q in range(mutate_num):
        x[mutate_idx[q]] = random.randint(-9,9)
    return x
    
def parallel_compute_aep(num):
    aep = floris_calc(df_wr,train_X[num,:])
    return aep

def constrain_function(x):
    distance = []
    x_cordination = []
    y_cordination = []
    turbine_num = int(len(x)/2)
    for i in range(turbine_num):
        x_cordination.append(x[2*i])
        y_cordination.append(x[2*i+1])
    for i in range(turbine_num):
        for j in range(i+1,turbine_num):
            distance1 = (x_cordination[i]-x_cordination[j])**2+(y_cordination[i]-y_cordination[j])**2
            distance1 = math.sqrt(distance1)
            distance.append(distance1)
    distance_min = min(distance)
    return (distance_min)
    

start_time = time.time()
#here input your wind rose
df_wr = pd.read_csv("windrose_case1.csv")

optim_sum = np.zeros((10,34))
#here define your layout variable bound
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

random_ini = random.sample(range(0,1000),10)
for k in range(10):    
    path = 'case1/switch/rand_ini_'+str(random_ini[k])
    os.makedirs(path)
for k in range(10):    
    mi_sampling = MixedIntegerSamplingMethod(LHS,design_space,criterion = 'ese', random_state = random_ini[k])
    
    num = 160

    train_X = mi_sampling(num)
    
    with multiprocessing.Pool(processes=32) as pool:
        train_Y = np.array(pool.map(parallel_compute_aep,range(num)))

    initial_set = np.zeros((num,33))
    for i in range(num):
        for j in range(32):
            initial_set [i][j] = train_X[i][j]
        initial_set [i][32] = train_Y[i]
    
    np.savetxt('case1/switch/rand_ini_'+str(random_ini[k])+'/initial.txt',initial_set,delimiter = ',',fmt='%.3f')    

    optimal_theta = [0.01]
    acq0 = 0.1
    iteration_times = 0
    init = mi_sampling(3000)
    for i in range(1000):
        f = open('case1/switch/rand_ini_'+str(random_ini[k])+'/records'+str(i+1)+'.txt','w')
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
        if population_var < 0.5:
            break
        population_x = population_x.reshape(-1,32)
        f.write('population is '+str(population_x)+'\n')
        f.write('aep is '+str(population_y)+'\n')
        f.write('variance of population is '+str(population_var)+'\n')
        
        surrogate_model = fit_model(train_X=train_X, train_Y=train_Y, theta0 = optimal_theta)
        init_predicted = surrogate_model.predict_values(np.array(init).reshape(-1,32))
        init_predicted = np.array(init_predicted).flatten()
        init_idx = np.argsort(init_predicted)[::-1]
        init_opt = []
        for j in range(500):
            init_opt = np.append(init_opt,init[init_idx[i]])
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
	#differential evolution implemented with scipy
        optimal_x, optimal_y, optimize_population_x = optimal_search(discrete_objective,bounds,design_space,'DE','discrete',20, sbo_seeds,nprocesses1,population_x,init_opt)
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
            bounds = discrete_mes.bounds
            infill_x = []
            mes_array = []
            for t in range(len(acq_seeds)):
                init_sampling = MixedIntegerSamplingMethod(LHS,design_space,criterion = 'ese')#,random_state = acq_seeds[t])
                init_all = init_sampling(1400)

                with multiprocessing.Pool(processes=60) as pool:
                    init_value = np.array(pool.map(parallel_compute_acq,range(1000)))
                init_value = np.array(init_value).flatten()
                init_index = np.argsort(init_value)[::-1]
                init_mes = []
                for m in range(60):
                    init_mes = np.append(init_mes,init_all[init_index[m]])
                    print (init_value[init_index[m]])
                init_mes = np.array(init_mes).reshape(-1,32)
                pop_mutate = []
                for m in range(10):
                    pop_mutate = np.append(pop_mutate,arr_mutate(population_x[i]))
                pop_mutate = pop_mutate.reshape(-1,32)
                init_mes = np.append(init_mes,pop_mutate,axis = 0)
                optimize_result = differential_evolution(discrete_mes,bounds,strategy='best2bin', maxiter=10000, popsize=70,init = init_mes,workers = 60,updating = 'deferred',mutation = (0.5,1.2),tol = 1e-15)

                decoded_solution = discrete_mes.decode(optimize_result.x)
                decoded_fun = discrete_mes(decoded_solution)
                infill_x = np.append(infill_x,decoded_solution)
                mes_array = np.append(mes_array,decoded_fun)
            infill_x = infill_x.reshape(-1,32)
            max_mes_index = np.argmin(mes_array)
            acq1 = min(mes_array)
            if -acq1<0.0005:
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
    
np.savetxt('case1/switch/optimal_dataset.csv',optim_sum,delimiter=',',fmt='%.3f')
