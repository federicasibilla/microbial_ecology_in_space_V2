"""
example_consumer_producer.py: script to run the spatial simulation of two species, where one consumes
the primary carbon source and the other consumes its byproduct

"""

import numpy as np
import pandas as pd

import cProfile
import pstats
import os
import sys

path = os.path.splitext(os.path.abspath(__file__))[0]
base_path = path.split('/microbial_ecology_in_space_V2')[0]
module_path = f'{base_path}/microbial_ecology_in_space_V2/models/shared'

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

from R_dynamics import *
from N_dynamics import *
from visualize_single_example import *
from create_matrices import *
from update import *
from SOR import *

# create paths to save results
path = os.path.splitext(os.path.abspath(__file__))[0]
executable_name = os.path.basename(__file__).split('.')[0]
results_dir = f"{path}_results"
graphs_dir = os.path.join(results_dir, "graphs")
matric_dir = os.path.join(results_dir, "matrices")
output_dir = os.path.join(results_dir, "outputs")
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(matric_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# initialize R0
n_r = 2
n_s = 2
n   = 40

# set random seed for reproducibility
np.random.seed(22)

# uptake matrix: A consumes PCS, B consumes what A produces
up_mat = np.array([[1.,0.],[0.,1.]])               

# defining sign matrix (all positive nutrients here)
sign_mat = np.ones((n_s,n_r))

# no essential nutrients (only catabolic cross-feeding)
mat_ess = np.zeros((n_s,n_r))

# only A produces secondary source
spec_met = np.array([[1.,1.],[1.,0.]])

# create metabolic matrix 
met_mat = np.array([[0.,0.],[1.,0.]])

# recapitulate in dictionary
mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met,
    'sign'    : sign_mat
}

# totally symmetric g and m
m = np.zeros((n_s))+0.1
g = np.array([1.,1.])
biomass = np.random.uniform(0, 2, (n, n))

# no reinsertion of chemicals
R0_wm = np.zeros((n_r))
R0_wm[0] = 1.
tau = np.zeros((n_r))+10

# initial guess for resources
guess = np.zeros((n_r))
guess[0]=10  

# define parameters
param = {
    # model parameters
    'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))*0.2,                         # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': R0_wm,                                      # used for 2D and wm   
    'tau' : tau,                                       # chemicals dilution 
    'alpha': 0.2,                                      # fixed partial uptake
    'guess_wm': guess,                                 # initial resources guess
    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.55,                                       # relaxation parameter
    'L'  : 10,                                         # grid true size        [length]
    'D'  : 10,                                         # diffusion constant    [area/time] 
    'Dz' : 0.001,                                      # ration between Dz and Dxy
    'acc': 1e-5,                                       # maximum accepted stopping criterion 
}

# initial guesses and conditions
R_space_ig = np.zeros((n,n,n_r))
R_space_ig[:,:,0]=0.5
N0_space   = encode(np.random.randint(0, n_s, size=(n,n)),np.array(np.arange(n_s)))
N0_wm      = np.bincount(decode(N0_space).flatten(), minlength=n_s)/(n*n)


# save in output file all parameters
with open(f"{output_dir}/parameters.txt", 'w') as file:
    file.write("Simulation Parameters:\n")
    for key, value in param.items():
        file.write(f"{key}: {value}\n")
    file.write("\nMatrices:\n")
    for key, value in mat.items():
        file.write(f"{key}:\n{value}\n\n")

last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, biomass  = simulate_3D_NBC(50000, f, growth_rates, R_space_ig, N0_space, biomass, param, mat)

# save results as csv
np.save(f'{output_dir}/R_fin.npz', current_R)
np.save(f'{output_dir}/N_fin.npz', current_N)
np.save(f'{output_dir}/2N.npz', last_2_frames_N)
np.save(f'{output_dir}/mod_fin.npz', mod)
np.save(f'{output_dir}/g_fin.npz', g_rates)
np.save(f'{output_dir}/shannon.npz', s_list)
np.save(f'{output_dir}/abundances.npz', abundances)

# save plots
R_ongrid(current_R,graphs_dir)
G_ongrid(g_rates,encode(last_2_frames_N[-2], np.array(np.arange(n_s))),graphs_dir)
N_ongrid(current_N,graphs_dir)
R_ongrid_3D(current_R,graphs_dir)
vis_abundances(abundances,s_list,graphs_dir)
#makenet(met_mat,matric_dir)
vispreferences(mat,matric_dir)
vis_intervals(t_list, graphs_dir)
