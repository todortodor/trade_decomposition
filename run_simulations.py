#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:04:22 2024

@author: slepot
"""

import utils as u
import numpy as np
import os

results_path='./results_more/'
try:
    os.mkdir(results_path)
except:
    pass

baseline = u.baseline()

baseline.make_np_arrays(inplace = True)
baseline.compute_shares_and_gammas(inplace = True)

carb_cost_list = np.concatenate([np.linspace(0,100,1001),
                                 np.linspace(100,1000,901)[1:]])
vec_init = None

for carb_cost in carb_cost_list:
    print(carb_cost)
    par = u.params(carb_cost=carb_cost)
    
    results = u.solve_one_loop(
                        params=par, 
                        baseline=baseline, 
                        vec_init = vec_init, 
                        tol=1e-9, 
                        damping=2)
    vec_init = np.concatenate([results['E_hat'].ravel(),
                               results['I_hat'].ravel(),
                               results['p_hat'].ravel()])

    u.write_solution_csv(results=results,
                         results_path=results_path,
                         run_name =str(round(carb_cost,1)),
                         params = par)

