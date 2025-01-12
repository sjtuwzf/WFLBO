import numpy as np
import random
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
from smt.utils.design_space import DesignSpace,IntegerVariable
import time

def constrained_sampling(bounds,num,seed,constrain):
    start_time = time.time()
    if seed is None:
        seed = random.randint(1,10000)
    print ('Initial sampling is beginning')
    mi_sampling = MixedIntegerSamplingMethod(LHS,bounds,criterion = 'cm',random_state = seed)
    x_init = mi_sampling(num)
    dim = x_init.shape[1]
    x_new = []
    j = 0 
    for i in range(x_init.shape[0]):
        if constrain(x_init[i]) == True:
            x_new = np.append(x_new,x_init[i])
    x_new = np.array(x_new).reshape(-1,dim)
    num1 = x_new.shape[0]
    #print ('Number of constrained initial samples is '+ str(num1))
    while num1 < num:
        j = j+1
        x_add_new = []
        num2 = num - num1
        if num2<2:
            num2 = 2
        x_add = mi_sampling.expand_lhs(x_new,num2,method = 'ese')
        for i in range(x_add.shape[0]):
            if constrain(x_add[i]) == True:
                x_add_new = np.append(x_add_new,x_add[i])
        x_add_new = np.array(x_add_new).reshape(-1,dim)
        x_new = np.append(x_new,x_add_new,axis = 0)
        num1 = x_new.shape[0]
        #print ('Number of added samples in '+str(j)+'th iteration is '+ str(x_add_new.shape[0]))
        print ('Number of constrained samples is '+ str(num1))
    end_time = time.time()
    last_time = end_time-start_time
    print ('Sampling time is '+ str(last_time) + ' s')
    x_new = x_new.reshape(-1,dim)
    return x_new


def constrained_sampling_DC(bounds,num,seed,constrain):
    start_time = time.time()
    bounds_num = len(bounds)
    random.seed(seed)
    random_state = random.sample(range(1000),bounds_num)
    x_init = [constrained_sampling(bounds[0],num,random_state[0],constrain)]
    dim = x_init[0].shape[1]
    for i in range(1,bounds_num):
        x_init0=constrained_sampling(bounds[i],num,random_state[i],constrain)
        x_init.append(x_init0)
        dim = dim+x_init0.shape[1]
    number = list(range(num))
    random.shuffle(number)
    idx = [number]
    for i in range(bounds_num):
        number = list(range(num))
        random.shuffle(number)
        idx.append(number)
    x_new = []
    for i in range(num):
        x_new0 = x_init[0][idx[0][i]]
        for j in range(1,bounds_num):
            x_new0 = np.append(x_new0,x_init[j][idx[j][i]])
        x_new = np.append(x_new,x_new0)
    x_new = x_new.reshape(-1,dim)
    end_time = time.time()
    last_time = end_time-start_time
    print ('Sampling time is '+ str(last_time) + ' s')
    return x_new
