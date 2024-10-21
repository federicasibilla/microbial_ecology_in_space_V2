"""
example_test_space.py: file to perform first few steps of spatial simulation with 8 species, 25 reaources

"""

import numpy as np
import pandas as pd

import cProfile
import pstats
import os
import sys

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


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
n_r = 8
n_s = 8
n   = 100

np.random.seed(22)

# drawing uptake matrix from binary distribution
up_mat = up_binary(n_s,n_r,3)


# random metabolic matrix: sample from Dir. + sparsity + constraints
met_mat = met_dir(n_r,0.8)

sign_mat = np.ones((n_s,n_r))

# no essential nutrients (only catabolic cross-feeding)
mat_ess = np.zeros((n_s,n_r))

# no auxotrophies (anyone can produce what metabolism allows)
spec_met = np.ones((n_s,n_r))

# totally symmetric g and m
g = np.ones((n_s))
m = np.zeros((n_s))
# initialize random biomass
biomass = np.random.uniform(0, 2, (n, n))

# no reinsertion of chemicals
R0 = np.zeros((n,n,n_r))
R0[:,:,0] = 10.
R0_wm = np.zeros((n_r))
R0_wm[0] = 10.
tau = np.zeros((n_r))+np.inf
tau[0]=1.

# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))*0.6,                         # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': R0_wm,                                      # used for 2D and wm   
    'tau' : tau,                                       # chemicals dilution                             
    'tau_s': 1.,                                       # species dilution
    'bb' : 2,                                          # index of the first building block +1

    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.55,                                       # relaxation parameter
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 10,                                         # diffusion constant    [area/time] 
    'Dz' : 0.1,                                        # ration between Dz and Dxy
    'acc': 1e-3,                                       # maximum accepted stopping criterion 
}

# define matrices dict
mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met,
    'sign'    : sign_mat
}

# initial guesses and conditions
R_space_ig = np.zeros((n,n,n_r))
R_space_ig[:,:,0]=5
N0_space   = encode(np.random.randint(0, 8, size=(n,n)),np.array(np.arange(n_s)))
N0_wm      = np.bincount(decode(N0_space).flatten(), minlength=n_s)/(n*n)


# save in output file all parameters
with open(f"{output_dir}/parameters.txt", 'w') as file:
    file.write("Simulation Parameters:\n")
    for key, value in param.items():
        file.write(f"{key}: {value}\n")
    file.write("\nMatrices:\n")
    for key, value in mat.items():
        file.write(f"{key}:\n{value}\n\n")

last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, biomass  = simulate_MG(10, f, growth_rates, R_space_ig, N0_space, biomass, param, mat,1)

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
makenet(met_mat,matric_dir)
vispreferences(mat,matric_dir)
vis_intervals(t_list, graphs_dir)

