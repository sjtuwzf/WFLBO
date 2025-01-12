import random
import numpy as np
#from optimization.opt_algorithm import Particle

            
class KRG_opt(object):
    def __init__(self,model):
        self.model = model
        
    def objective_function(self,x):
        return self.model.predict_values(np.array(x).reshape(-1,4))
            
    def PSO(self, num_particles, num_iterations, dim, lb, ub):
        particles = [Particle(self.model, dim, lb, ub) for _ in range(num_particles)]
        gbest = max(particles, key=lambda p: self.objective_function(p.position)).position

        for _ in range(num_iterations):
            for particle in particles:
                particle.update(gbest)
                if self.objective_function(particle.position) > self.objective_function(gbest):
                    gbest = particle.position.copy()
                
        return gbest, self.objective_function(gbest)
    
class Particle:
    def __init__(self, model, dim, lb, ub):
        self.model = model
        self.position = [random.uniform(l, u) for l, u in zip(lb, ub)]
        self.pbest = self.position.copy()
        self.velocity = []
    def update_velocity(self, gbest):
        self.velocity.clear()
                # Clears the current velocity of the particle
        for i in range(len(self.position)):
            if self.position[i] in gbest:
                j = gbest.index(self.position[i])
                self.velocity.append((i, j))
                self.position[i], self.position[j] = self.position[j], self.position[i]
    def move(self):
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]
    def update(self, gbest):
        self.update_velocity(gbest)
        self.move()
        if self.objective_function(self.position) > self.objective_function(self.pbest):
            self.pbest = self.position.copy()
    def objective_function (self,x):
        return self.model.predict_values(np.array(x).reshape(-1,4))
    
    
#def KRG_predict(model,x)
                
def continuous_optimization(model,design_space,opt_method,opt_num):
    optimal_value = []
    optimal_location = []
    
    bounds = design_space
    dim = bounds.shape [0]
    lb = bounds[:,0].tolist()
    ub = bounds[:,1].tolist()
    
    Optimal_Sampling = KRG_opt(model)
    
    if opt_method == 'PSO':
        for i in range(opt_num):
            optimal_x, optimal_y = Optimal_Sampling.PSO(50, 100, dim, lb, ub)
            optimal_value.append(optimal_y)
            optimal_location.append(optimal_x)
        optimal_value = np.array(optimal_value)
        optimal_location = np.array(optimal_location)
    else:
        print('This optimization is not included in this toolkit. Please use PSO')
    return optimal_location, optimal_value