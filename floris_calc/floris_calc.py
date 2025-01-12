import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from floris.tools import FlorisInterface
from floris.tools.layout_functions import visualize_layout

def floris_calc(wr,layout):
    # wr is the wind rose
    # layout set is a set of variables
    wd_array = np.array(wr["wd"].unique(), dtype=float)
    ws_array = np.array(wr["ws"].unique(), dtype=float)

    wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
    freq_interp = NearestNDInterpolator(wr[["wd", "ws"]], wr["freq_val"])
    freq = freq_interp(wd_grid, ws_grid)

    freq = freq / np.sum(freq)
    
    turbine_num = int(len(layout)/2)
    coordination_x = np.zeros(turbine_num)
    coordination_y = np.zeros(turbine_num)
    
    for i in range(0,turbine_num):
        coordination_x[i] = layout[2*i]
        coordination_y[i] = layout[2*i+1]        
    layout_x = []
    layout_y = []
    layout_x = np.append(layout_x,coordination_x[0])
    layout_y = np.append(layout_y,coordination_y[0])
    for i in range(1,len(coordination_x)):
        distance = []
        for j in range(0,len(layout_x)):
            distance1 = abs(layout_x[j]-coordination_x[i])+abs(layout_y[j]-coordination_y[i])
            distance = np.append(distance,distance1)
        if min(distance) != 0:
            layout_x = np.append(layout_x,coordination_x[i])
            layout_y = np.append(layout_y,coordination_y[i])
            
    fi = FlorisInterface("floris_calc/inputs/gch.yaml") 

    D = fi.floris.farm.rotor_diameters[0]
    
    layout_x = layout_x*D
    layout_y = layout_y*D

    fi.reinitialize(
        layout_x=layout_x,
        layout_y=layout_y,
        wind_directions=wd_array,
        wind_speeds=ws_array,
    )

    aep = fi.get_farm_AEP(
        freq=freq,
        cut_in_wind_speed=3.0,  # Wakes are not evaluated below this wind speed
        cut_out_wind_speed=25.0,  # Wakes are not evaluated above this wind speed
    )
    
    aep = aep/(1.0e9)
    
    return aep