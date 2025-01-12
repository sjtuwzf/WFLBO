import numpy as np
import math
from scipy.stats import norm
#This python file helps calculate max value entropy search acqusition function for Kriging built from smt.


class MaxValueEntropySearch (object):
    
    def __init__(self,GPmodel,bounds,maximum):
        self.GPmodel = GPmodel
        self.bounds = bounds
        self.maximum = maximum
                    
    def mes(self, x):
        acq = 0
        means = self.GPmodel.predict_values (x)
        variances = self.GPmodel.predict_variances (x)     
        std = np.sqrt(variances)
        for i in range(self.maximum.shape[0]):
            normalized_max = (self.maximum[i]-means)/std
            pdf1 = norm.pdf(normalized_max)
            cdf1 = norm.cdf(normalized_max)
            if cdf1 < 1e-30:
                cdf1 = 1e-30
            arg1 = normalized_max*pdf1/(2*cdf1)
            arg2 = np.log(cdf1)
            acq = acq+arg1-arg2
            acq = acq[0][0]
            
        acq = acq/self.maximum.shape[0]
        if variances < 1e-10:
            acq = 0
      
        return acq
    
