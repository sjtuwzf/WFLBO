import random
import numpy as np
import multiprocessing
from scipy.optimize import differential_evolution, dual_annealing
import multiprocessing
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod,MixedIntegerContext
import pygad

class discrete_optimization(object):
    def __init__(self,objective_function,bounds,design_space,opt_method,opt_nums,opt_num_seeds,x0,init):
        self.objective_function=objective_function
        self.bounds = bounds
        self.design_space = design_space
        self.opt_method = opt_method
        self.opt_nums = opt_nums
        self.opt_num_seeds = opt_num_seeds
        self.x0 = x0
        self.init = init
        self.dim = init.shape[1]
        
    def optimize_parallel(self,njobs):
        seeds = self.opt_num_seeds
        optimal_value = []
        optimal_location = []
        last_population = []
        if self.opt_method == 'PSO':
            with multiprocessing.Pool(processes=njobs) as pool:
                optimize_results = pool.starmap(self.pso_optimize, [(seed,) for seed in seeds])
        elif self.opt_method == 'DE':
            with multiprocessing.Pool(processes=njobs) as pool:
                optimize_results = pool.starmap(self.de_optimize, [(seed,) for seed in seeds])
        elif self.opt_method == 'dual_annealing':
            with multiprocessing.Pool(processes=njobs) as pool:
                optimize_results = pool.starmap(self.dual_annealing_optimize, [(seed,) for seed in seeds])
        elif self.opt_method == 'GA':
            with multiprocessing.Pool(processes=njobs) as pool:
                optimize_results = pool.starmap(self.ga_optimize, [(seed,) for seed in seeds])
        else:
            print('This optimization is not included in this toolkit. Please use PSO or differential evolution method.')
            
        for i in range(self.opt_nums):
            arr0 = np.array(optimize_results[i][0])
            optimal_location.append(arr0)
            optimal_value = np.append(optimal_value,optimize_results[i][1])
            #last_population.append(optimize_results[i][2])
            #last_population_fun = np.append(last_population_fun,optimize_results[i][3])
        optimal_location = np.array(optimal_location)
        if self.opt_method == 'dual_annealing':
            return optimal_location, optimal_value, None
        else:
            for i in range(self.opt_nums):
                last_population.append(optimize_results[i][2])
            last_population = np.array(last_population)
            return optimal_location, optimal_value, last_population#, last_population_fun
            
    def de_optimize(self,seed):
        if self.x0 is None:
            optimize_result = differential_evolution(self.objective_function,self.bounds,strategy='best1bin', maxiter=10000, popsize=50,seed = seed, mutation = 0.7,workers = 12)
        else:
            if self.init is None:
                mi_sampling = MixedIntegerSamplingMethod(LHS,self.design_space,criterion = 'ese', random_state = seed)
                init = mi_sampling(50)
            #init0_value = surrogate_model.predict_values(np.array(init_all).reshape(-1,32))
                init = np.append(init,self.x0,axis = 0)
                optimize_result = differential_evolution(self.objective_function,self.bounds,strategy='best1bin', maxiter=10000, popsize=70, mutation = (0.5,1.2), init = init,tol = 1e-5)
            else:
                random.seed(seed)
                init_idx = random.sample(range(self.init.shape[0]),70)
                x0_idx = random.sample(range(20),3)
                init0 = []
                #x0_ = []
                for i in range(70):
                    init0 = np.append(init0,self.init[init_idx[i]])
                #for i in range(3):
                #    x0_ = np.append(x0_,self.x0[x0_idx[i]])
                #init = np.append(init0,self.x0)
                init = init0.reshape(-1,self.dim)
                optimize_result = differential_evolution(self.objective_function,self.bounds,strategy='best1bin', maxiter=10000, popsize=70, mutation = (0.5,1.2), init = init,tol = 1e-5)
        dim = len(optimize_result.x)
        decoded_population = []
        for i in range(70):            
            decoded_population = np.append(decoded_population,self.objective_function.decode(optimize_result.population[i]))
        decoded_population = decoded_population.reshape(-1,dim)
        #decoded_population_fun = self.objective_function(decoded_population)
        decoded_solution = self.objective_function.decode(optimize_result.x)
        decoded_fun = self.objective_function(decoded_solution)
        return decoded_solution,decoded_fun, decoded_population#, decoded_population_fun
    
    def pso_optimize(self,seed):
        optimal_x, optimal_y = pso(self.objective_function,self.bounds)
        return optimal_x, optimal_y
    
    def dual_annealing_optimize(self,seed):
        random.seed(seed)
        random_state = random.randint(0,self.opt_nums-1)
        optimize_result = dual_annealing(self.objective_function,self.bounds, maxiter=1000, seed = seed, x0 = self.x0[0])#[random_state])
        decoded_solution = self.objective_function.decode(optimize_result.x)
        decoded_fun = self.objective_function(decoded_solution)
        return decoded_solution,decoded_fun

    def ga_optimize(self,seed):
        if self.x0 is None:
            GA_optimize = pygad.GA(num_generations = 1000, num_parents_mating = 50, fitness_func = self.objective_function, gene_space = self.bounds, gene_type = int, random_seed = seed)
        else:
            mi_sampling = MixedIntegerSamplingMethod(LHS,self.design_space,criterion = 'ese', random_state = seed)
            init = mi_sampling(30)
            init = np.append(init,self.x0,axis = 0)
            GA_optimize = pygad.GA(num_generations = 1000, num_parents_mating = 50, fitness_func = self.objective_function, gene_space = self.bounds, gene_type = int, initial_population = init)
        GA_optimize.run()
        solution, solution_fitness, solution_idx = GA_optimize.best_solution(GA_optimize.last_generation_fitness)
        return solution, solution_fitness


def pso(objective_function, bounds, info=False):
    dim = bounds.shape[0]
    lb = bounds[:, 0].tolist()
    ub = bounds[:, 1].tolist()
    # Initialization
    ratio1 = 0.2
    ratio2 = 0.2
    noP=20
    Max_iteration = 10000
    Vmax=6
    wMax=0.9
    wMin=0.2
    c1=2
    c2=2
    class Particle:
        def __init__(self, dim, lb, ub):
            self.X = [random.randint(l, u) for l, u in zip(lb, ub)]
            self.PBEST_X = self.X.copy()
            self.PBEST_O = np.inf
            self.V = np.random.rand(dim)

    Swarm = []
    for _ in range(noP):
        Swarm.append(Particle(dim, lb, ub))
    # for particle in Swarm:
    #     print(f"Particle X: {particle.X}")
        
    # Global best initialization
    GBEST_X = np.zeros(dim)
    GBEST_O = np.inf

    # Variables for consecutive same best check
    previous_best_x = np.zeros(dim)
    consecutive_same_best_count = 0
    consecutive_same_best_limit = 3

    for t in range(Max_iteration):
        # Calculate objective function and update personal best and global best
        for k in range(noP):
            Swarm[k].O = objective_function(Swarm[k].X)
            if Swarm[k].O < Swarm[k].PBEST_O:
                Swarm[k].PBEST_O = Swarm[k].O
                Swarm[k].PBEST_X = Swarm[k].X.copy()

            if Swarm[k].O < GBEST_O:
                GBEST_O = Swarm[k].O
                GBEST_X = Swarm[k].X.copy()
        # Check for consecutive same best if t > int(Max_iteration * ratio2)
        if t > int(Max_iteration * ratio2):
            GBEST_X = np.round(GBEST_X).astype(int)
            GBEST_O = objective_function(GBEST_X)
            if np.array_equal(GBEST_X, previous_best_x):
                consecutive_same_best_count += 1
            else:
                consecutive_same_best_count = 0

            # Update previous best
            previous_best_x = GBEST_X.copy()

            # Stop convergence if consecutive same best limit is reached
            if consecutive_same_best_count >= consecutive_same_best_limit:
                if info:
                    print(f"Convergence achieved: {consecutive_same_best_limit} consecutive same best.")
                break

        # Check for rounding to integers if t > int(Max_iteration * ratio)
        if t > int(Max_iteration * ratio1):
            for k in range(noP):
                Swarm[k].X = [int(x) for x in Swarm[k].X]
                

        # Update inertia weight
        w = wMax - t * ((wMax - wMin) / Max_iteration)

        # Update velocity and position
        for k in range(noP):
            # first_vel = Swarm[k].V.copy()
            Swarm[k].V = (w * np.array(Swarm[k].V) +
                c1 * np.random.rand(dim) * (np.array(Swarm[k].PBEST_X) - np.array(Swarm[k].X)) +
                c2 * np.random.rand(dim) * (GBEST_X - np.array(Swarm[k].X)))
    

            # Apply velocity limits
            Swarm[k].V[Swarm[k].V > Vmax] = Vmax * np.random.rand()
            Swarm[k].V[Swarm[k].V < -Vmax] = -Vmax * np.random.rand()

            # Update position and handle position limits
            # first_loc = Swarm[k].X.copy()
            Swarm[k].X = Swarm[k].X + Swarm[k].V

            # Handle position limits
            Swarm[k].X = [min(x, u) if x > u else x for x, u in zip(Swarm[k].X, ub)]
            Swarm[k].X = [max(x, l) if x < l else x for x, l in zip(Swarm[k].X, lb)]
        if info:
            print(f"Iteration {t + 1}, Current Best Objective Value: {GBEST_O} at {GBEST_X}")
    GBEST_Xint = np.round(GBEST_X).astype(int)
    GBEST_O = objective_function(GBEST_Xint)
    if info:
        print(f"Iteration {t + 1}, Current Best Objective Value Int: {GBEST_O} at {GBEST_Xint}")

    # Create a dictionary to store the results
    result = {
        'message': 'Optimization terminated successfully.',
        'success': True,
        'fun': GBEST_O,
        'x': GBEST_Xint.tolist(),
        'nit': t + 1,
        'nfev': (t + 1) * noP,  # Assuming each particle evaluation is counted
    }

    return GBEST_Xint.tolist(), GBEST_O
