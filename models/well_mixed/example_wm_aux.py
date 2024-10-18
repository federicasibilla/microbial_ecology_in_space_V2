"""
example_wm_aux.py: file to perform well-mixed simulations of two auxotrophs

"""

import sys
import os
import pickle

path = os.path.splitext(os.path.abspath(__file__))[0]
base_path = path.split('/microbial_ecology_in_space')[0]
module_path = f'{base_path}/microbial_ecology_in_space/models/shared'

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

from visualize import *
from well_mixed import *

# directory for results
results_dir = f"{path}_results"
os.makedirs(results_dir, exist_ok=True)

n_r = 3
n_s = 2

# make matrices
up_mat   = np.array([[1.,0.,1.],[1.,1.,0.]])
met_mat  = np.array([[0.,0.,0.],[1.,0.,0.],[1,0.,0.]])
sign_mat = np.array([[1.,0.,0.],[1.,0.,0.]])
mat_ess  = np.array([[0.,0.,1.],[0.,1.,0.]])
spec_met = np.array([[0.,1,0.],[0.,0.,1]])

# totally symmetric g and m
m = np.zeros((n_s))+0.1

# no reinsertion of produced chemicals
ext = np.zeros((n_r))
ext[0] = 10.
tau = np.zeros((n_r))+10

g = np.array([0.5,1.])

# initial abundance and initial R guess
N0 =np.array([0.1,0.1])
guess = np.array([10,10,10])

# define parameters
param = {
    # model parameters
    'w'  : np.ones((n_r))*10,                          # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))*0.2,                         # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': ext,                                        # external replenishment  
    'tau' : tau,                                       # chemicals dilution                             
    #'tau_s': 0.1,                                      # species dilution
    'guess_wm': guess,                                 # initial resources guess
    'alpha': 0.2                                       # partial modulation parameter                      
}

# define matrices dict
mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met,
    'sign'    : sign_mat
}

N_fin,R_fin=run_wellmixed(N0,param,mat,dR_dt_partial,dN_dt,2000)

sums = np.sum(N_fin, axis=1)[-1]
final_fraction = np.where(sums == 0, 0, N_fin[-1] / (sums+1e-15))
print('final_fraction: ', final_fraction)

print('final_abundances: ', N_fin[-1], R_fin[-1])

vis_wm(N_fin,R_fin[1:],results_dir)