import numpy as np
from optimization import discrete_optimization as do
from optimization import continuous_optimization as co

def optimal_search(objective_function,bounds,design_space,opt_method,opt_type,opt_num,opt_num_seeds,nprocesses,x0=None,init=None):
    #model: the model to be optimized
    #design_space: lower bounds and upper bounds are defined
    #opt_method: Particle swarm optimization (PSO) is the only optimization method option currently
    #opt_type: 'continuous_optimization' or 'discrete_optimization'
    #opt_num: number of optimal solution
    if opt_type == 'discrete':
        do_opt = do.discrete_optimization(objective_function,bounds,design_space,opt_method,opt_num,opt_num_seeds,x0,init)
        optimal_x, optimal_y, population_x = do_opt.optimize_parallel(nprocesses)
    else:
        optimal_x, optimal_y, population_x = co.continuous_optimization(objective_function,bounds,design_space,opt_method,opt_num,opt_num_seeds,x0,init)
    return optimal_x, optimal_y, population_x


    
    