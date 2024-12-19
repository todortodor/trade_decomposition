#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:13:40 2024

@author: dora
"""

import utils as u
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

results_folder = 'results_more'

baseline = u.baseline()

baseline.make_np_arrays(inplace = True)
baseline.compute_shares_and_gammas(inplace = True)

# carb_cost_list = np.concatenate([np.linspace(0,100,101),
#                                  np.linspace(100,1000,91)[1:]])
carb_cost_list = np.concatenate([np.linspace(0,100,1001),
                                 np.linspace(100,1000,901)[1:]])

sols = []
for carb_cost in carb_cost_list:
    print(carb_cost)
    # sol = u.sol(results_path=f'{results_folder}/{str(round(carb_cost))}_results.csv', 
    sol = u.sol(results_path=f'{results_folder}/{str(round(carb_cost,1))}_results.csv', 
                baseline=baseline,
                carb_cost=carb_cost)
    sol.compute_solution(baseline)
    sols.append(sol)
    
S = baseline.sector_number
N = baseline.country_number

#%% 

def Q_hat(sol,baseline):
    res = np.einsum('is,is->',
                    sol.q_hat.value.values.reshape(N,S),
                    baseline.output_np
                    )
    return res

def Q_s_hat(sol,baseline):
    res = np.einsum('is,is->s',
                    sol.q_hat.value.values.reshape(N,S),
                    baseline.output_np
                    )
    return res

def Q_is_hat(sol,baseline):
    res = sol.q_hat.value.values.reshape(N,S)
    return res

def E(sol):
    res = sol.co2_prod.value.sum()
    return res

def E_s(sol):
    res = sol.co2_prod.value.groupby('sector').sum().values
    return res

def E_is(sol):
    res = sol.co2_prod.value.values.reshape(N,S)
    return res

l_dterm_1 = []
l_dterm_2 = []
l_dterm_3 = []
l_dE = []

for i in range(len(sols)-1):
    print(i)
    sol_baseline = sols[i]
    sol_cf = sols[i+1]
    
    term_1 = E(sol_baseline)*(Q_hat(sol_cf,baseline)/Q_hat(sol_baseline,baseline) - 1)

    term_2 = np.einsum('s,s->',
        E_s(sol_baseline),
        Q_s_hat(sol_cf,baseline)*Q_hat(sol_baseline,baseline)
            /(Q_hat(sol_cf,baseline)*Q_s_hat(sol_baseline,baseline)) - 1
        )
    
    term_3 = np.einsum('is,is->',
        E_is(sol_baseline),
        Q_is_hat(sol_cf,baseline)*Q_s_hat(sol_baseline,baseline)
            /(Q_s_hat(sol_cf,baseline)*Q_is_hat(sol_baseline,baseline)) - 1
        )
    
    dE = sol_cf.co2_prod.value.sum() - sol_baseline.co2_prod.value.sum()
    
    l_dterm_1.append(term_1)
    l_dterm_2.append(term_2)
    l_dterm_3.append(term_3)
    l_dE.append(dE)
    
l_term_1 = np.array([sum(l_dterm_1[:i]) for i in range(len(l_dterm_1)+1)])
l_term_2 = np.array([sum(l_dterm_2[:i]) for i in range(len(l_dterm_2)+1)])
l_term_3 = np.array([sum(l_dterm_3[:i]) for i in range(len(l_dterm_3)+1)])
l_deltaE = np.array([sum(l_dE[:i]) for i in range(len(l_dE)+1)])

term_labels = {
    'term_1':'Scale',
    'term_2':'Composition',
    'term_3':'Green sourcing'
          }
#%%

fig,ax = plt.subplots(1,2,figsize = (24, 12))

ax[0].stackplot(carb_cost_list,
              [term/baseline.co2_prod.value.sum() for term in [l_term_1,l_term_2,l_term_3]],
              labels=[term_labels[term] for term in term_labels.keys()])
ax[0].plot(carb_cost_list,l_deltaE/baseline.co2_prod.value.sum(), 
          label='Emissions',color='black'
          ,lw=3)
ax[0].legend(loc='lower left',fontsize = 20, frameon=True, fancybox=True)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[0].set_xlabel('Carbon tax ($/ton of CO2eq.)',fontsize = 20)

sum_terms = l_term_1+l_term_2+l_term_3

y = [-term/sum_terms for term in [l_term_1,l_term_2,l_term_3]]

ax[1].stackplot(carb_cost_list,y,
              labels=[term_labels[term] for term in term_labels.keys()])

offset = -0.005
for i,term_label in enumerate(term_labels.values()):
    term = [l_term_1,l_term_2,l_term_3][i]
    loc = 1450
    ax[1].text(carb_cost_list[loc], 
               -(term[loc]/sum_terms[loc])/2+offset, 
               term_label+' : '+str(((term[999]/sum_terms[999]).mean()*100).round(1))+'%',
               ha='center', va='center',color='black',fontsize = 20)
    offset = offset-(term[1:]/sum_terms[1:])[loc]
    
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].set_xlabel('Carbon tax ($/ton of CO2eq.)',fontsize = 20)

plt.show()