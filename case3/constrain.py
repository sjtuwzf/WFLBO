import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

class constrain_func(object):
    
    def __init__(self,x):
        self.x = x*126
    def data_prepare(self):
        boundaries_point = np.array([
            [10363.8, 6490.3], [9449.7, 1602.2], [9387.0, 1056.6], [9365.1, 625.5], 
            [9360.8, 360.2], [9361.5, 126.9], [9361.3, 137.1], [7997.6, 1457.9], 
            [6098.3, 3297.5], [8450.3, 6455.3], [8505.4, 6422.3], [9133.0, 6127.4], 
            [9332.8, 6072.6], [9544.2, 6087.1], [9739.0, 6171.2], [9894.9, 6316.9], 
            [10071.8, 6552.5], [10106.9, 6611.1]
        ])
        x0 = []
        y0 = []

        for i in range(len(boundaries_point)):
            x0 = np.append(x0, boundaries_point[i][0])
            y0 = np.append(y0, boundaries_point[i][1])
        x0 = x0 - 8000
        y0 = y0 - 3000

        x_up = []
        y_up = []
        for i in range(7):
            x_up = np.append(x_up, x0[i+11])
            y_up = np.append(y_up, y0[i+11])
        self.x_up = x_up
        self.y_up = y_up

        x_down = []
        y_down = []
        for i in range(6):
            x_down = np.append(x_down, x0[i+1])
            y_down = np.append(y_down, y0[i+1])
        self.x_down = x_down
        self.y_down = y_down
        coeffs1 = np.array([1.10106627e-3,-3.08527114e0,5.21753355e3])
        coeffs2 = np.array([-2.71181334e-01,7.77322220e+02,-5.58361736e+05])
        xl1 = [-1901.7, 450.3]
        yl1 = [297.5, 3455.3]
        xl2 = [450.3, 1133]
        yl2 = [3455.3, 3127.4]
        xl3 = [2106.9, 2363.8]
        yl3 = [3611.1, 3490.3]
        xl4 = [2363.8, 1449.7]
        yl4 = [3490.3, -1397.8]
        xl5 = [1361.3, -1901.7]
        yl5 = [-2862.9, 297.5]
        
        return coeffs1,coeffs2
    
    def constrain_quadratic_func1(self,turbine_coordinate,coeffs):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        y = quadratic(turbine_x,*coeffs)
        if turbine_y>y:
            condition = False
        if turbine_y == 0 :
            turbine_y = 1e-10
        return condition,y/turbine_y    
    
    def constrain_quadratic_func2(self,turbine_coordinate,coeffs):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        y = quadratic(turbine_x,*coeffs)
        if turbine_y<y:
            condition = False
        return condition,turbine_y/y 
    
    def constrain_linear_func1(self,turbine_coordinate):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        xl = [-1901.7, 450.3]
        yl = [297.5, 3455.3]
        k = (yl[1]-yl[0])/(xl[1]-xl[0])
        b = yl[1]-k*xl[1]
        y = k*turbine_x+b
        if turbine_y>y:
            condition = False
        if turbine_y == 0 :
            turbine_y = 1e-10
        return condition,y/turbine_y
        
    def constrain_linear_func2(self,turbine_coordinate):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        xl = [450.3, 1133]
        yl = [3455.3, 3127.4]
        k = (yl[1]-yl[0])/(xl[1]-xl[0])
        b = yl[1]-k*xl[1]
        y = k*turbine_x+b
        if turbine_y>y:
            condition = False
        if turbine_y == 0 :
            turbine_y = 1e-10
        return condition,y/turbine_y
    
    def constrain_linear_func3(self,turbine_coordinate):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        xl = [2106.9, 2363.8]
        yl = [3611.1, 3490.3]
        k = (yl[1]-yl[0])/(xl[1]-xl[0])
        b = yl[1]-k*xl[1]
        y = k*turbine_x+b
        if turbine_y>y:
            condition = False
        if turbine_y == 0 :
            turbine_y = 1e-10
        return condition,y/turbine_y
    
    def constrain_linear_func4(self,turbine_coordinate):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        xl = [2363.8, 1449.7]
        yl = [3490.3, -1397.8]
        k = (yl[1]-yl[0])/(xl[1]-xl[0])
        b = yl[1]-k*xl[1]
        y = k*turbine_x+b
        if turbine_y<y:
            condition = False
        return condition,turbine_y/y
    
    def constrain_linear_func5(self,turbine_coordinate):
        condition = True
        turbine_x = turbine_coordinate[0]
        turbine_y = turbine_coordinate[1]
        xl = [1361.3, -1901.7]
        yl = [-2862.9, 297.5]
        k = (yl[1]-yl[0])/(xl[1]-xl[0])
        b = yl[1]-k*xl[1]
        y = k*turbine_x+b
        if turbine_y<y:
            condition = False
        return condition,turbine_y/y
        
    def constrain(self):
        turbines_coordinations = self.x
        turbines_coordinate = np.array(turbines_coordinations).reshape(-1,2)
        turbines_num = turbines_coordinate.shape[0]
        conditions = []

        coeffs1,coeffs2 = self.data_prepare()
        for i in range(turbines_num):
            conditions1,y1 = self.constrain_quadratic_func1(turbines_coordinate[i],coeffs1)
            conditions2,y2 = self.constrain_quadratic_func2(turbines_coordinate[i],coeffs2)
            conditions3,y3 = self.constrain_linear_func1(turbines_coordinate[i])
            conditions4,y4 = self.constrain_linear_func2(turbines_coordinate[i])
            conditions5,y5 = self.constrain_linear_func3(turbines_coordinate[i])
            conditions6,y6 = self.constrain_linear_func4(turbines_coordinate[i])
            conditions7,y7 = self.constrain_linear_func5(turbines_coordinate[i])
            conditions = [conditions1,conditions2,conditions3,conditions4,conditions5,conditions6,conditions7]
            y_ = [y1,y2,y3,y4,y5,y6,y7]
            y_new = [1 if x>=1 else x for x in y_]
            if all(conditions):
                condition = True
            else:
                condition = False
            if condition == False:
                break
        coeff_y = 1
        for j in range(7):
            coeff_y = coeff_y * y_new[j]
        return condition, coeff_y
    
    
