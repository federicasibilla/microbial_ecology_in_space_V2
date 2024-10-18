"""
example_wm_825.py: file to perform the simulation of 8 species with 24 shared metabolites and one externally supplied carbon source

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

# initialize R0
n_r = 25
n_s = 8
n   = 100

np.random.seed(43)

# drawing uptake matrix from binary distribution
up_mat = np.zeros((n_s,n_r))
# each species has a given number of preferred resources
for i in range(n_s):
    ones_indices = np.random.choice(n_r, 5, replace=False)
    up_mat[i, ones_indices] = 1.0
# check that someone eats primary source
if (up_mat[:,0] == 0).all():
    up_mat[np.random.randint(0, n_s-1),0] = 1.0

# random metabolic matrix: sample from Dir. + sparsity + constraints
met_mat = np.ones((n_r,n_r))*(np.random.rand(n_r, n_r) > 0.8)      # make metabolic matrix sparce
met_mat[0,:] = 0                                                   # carbon source is not produced
np.fill_diagonal(met_mat, 0)                                       # diagonal elements should be 0
# check that at least one thing is produced from primary carbon source
if (met_mat[:,0] == 0).all():
    met_mat[np.random.randint(0, n_r-1),0] = 1
# sample all from D. distribution
for column in range(n_r):
    # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
    non_zero_indices = np.where(met_mat[:, column] == 1)[0]  
    if len(non_zero_indices) > 0:
        # Sample from Dirichlet distribution for non-zero entries
        dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
        met_mat[non_zero_indices, column] = dirichlet_values

sign_mat = np.ones((n_s,n_r))

# no essential nutrients (only catabolic cross-feeding)
mat_ess = np.zeros((n_s,n_r))

# no auxotrophies (anyone can produce what metabolism allows)
spec_met = np.ones((n_s,n_r))

# totally symmetric g and m
g = np.ones((n_s))
m = np.zeros((n_s))+0.1

# no reinsertion of chemicals
tau = np.zeros((n_r))+ 10
ext = np.zeros((n_r))
ext[0] = 10
guess = np.ones((n_r))*10

# define parameters
param = {
    # model parameters
    'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))*0.8,                         # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': ext,                                        # external replenishment  
    'tau' : tau,                                       # chemicals dilution                             
    'tau_s': 10,                                       # species dilution
    'guess_wm': guess                                  # initial resources guess
}



# define matrices dict
mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met,
    'sign'    : sign_mat
}
print(up_mat)

N_fin,R_fin=run_wellmixed(np.ones((n_s))*0.1,param,mat,dR_dt_maslov,dN_dt_maslov,2000)

vis_wm(N_fin,R_fin[1:],results_dir)
vispreferences(mat,results_dir)