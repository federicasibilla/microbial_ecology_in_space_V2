"""

CR_generation.py: script to generate a large set of consumer resource models across a wide range of parameters, and to run
                     them to equilibrium to then map them onto LV models
"""

import os
import sys

path = os.path.abspath('')
base_path = path.split('/microbial_ecology_in_space')[0]
module_path = f'{base_path}/microbial_ecology_in_space/models/shared'
module_path_1 = f'{base_path}/microbial_ecology_in_space/models/well_mixed'

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path_1 not in sys.path:
    sys.path.append(module_path_1)

# libraries imports

import pickle
import numpy as np

from scipy.integrate import solve_ivp

import visualize
import definitions
import well_mixed

#-------------------------------------------------------------------------------------------------------------
# functions for the mapping
# functions definition 

def f_i_alpha(R,param,mat):
    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters
    mat: dictionary, matrices

    returns matrix, n_sxn_r, the impact function of species i on resource alpha in a matrix

    """

    n_s = len(param['g'])
    n_r = len(R)

    fialpha = np.zeros((n_s,n_r))

    for i in range(n_s):
        for alpha in range(n_r):
            f_ia = -mat['uptake'][i,alpha]*R[alpha]/(1+R[alpha])+1/param['w'][alpha]*np.sum((param['w']*param['l']*mat['uptake'][i]*R/(1+R))*mat['met'][alpha])
            fialpha[i,alpha]+=f_ia

    return fialpha

def sigma(R,param):
    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters

    returns vector, n_r, replenishment function for each resource

    """
    sigma = (param['ext']-R)/param['tau']

    return sigma

def grad_i_alpha(R,param,mat):
    """
    R: vector, n_r, contains concentrations
    param: dictionary, parameters
    mat: dictionary, matrices

    returns matrix, n_sxn_r, containing the alpha component of the gradient of the sensitivity function of i

    """
    n_s = len(param['g'])
    n_r = len(R)

    gialpha = np.zeros((n_s,n_r))

    for i in range(n_s):
        for alpha in range(n_r):
            g_ia = param['g'][i]*param['w'][alpha]*mat['uptake'][i,alpha]*(1-param['l'][alpha])*1/(1+R[alpha])**2
            gialpha[i,alpha]=g_ia

    return gialpha

# simulate LV with these parameters

def LV_model(t,y,r0,A,n_s):

    sp_abund = y

    # Set abundances to zero if they are less than a threshold
    #sp_abund[sp_abund < 1e-10] = 0

    dsdt = [sp_abund[alpha]*(r0[alpha] + np.dot(A[alpha,:],sp_abund)) for alpha in range(n_s)]
    
    return dsdt
   
#---------------------------------------------------------------------------------------------------------------

def CR_and_mapping(n_resources,supplied,average_consumed,leakage,replica):

    """
    n_resources: int, number of total resources
    supplied: int, number of resources that are externally supplied
    average_consumed: int, number of resources that are consumed by each species
    leakage: float, leakage parameter

    RETURNS: a dictionary with the parameters and a pickle file with all the stored data

    """

    # create paths to save results
    path = os.path.splitext(os.path.abspath(__file__))[0]
    executable_name = os.path.basename(__file__).split('.')[0]
    results_dir = f"{path}_results/{n_resources}_{supplied}_{average_consumed}_{leakage}_{replica}"
    os.makedirs(results_dir, exist_ok=True)

    n_resources = int(n_resources)
    supplied = int(supplied)
    average_consumed = int(average_consumed)
    replica = int(replica)

    np.random.seed(replica)

    # number of species and number of nutrients

    n_s = 8
    n_r = n_resources

    # functions used in this sections are in the shared/definitions.py file

    # defining binary uptake matrix with 5 preferences for each species
    up_mat = definitions.up_binary(n_s,n_r,average_consumed)                

    # defining sign matrix (all positive nutrients here)
    sign_mat = np.ones((n_s,n_r))

    # no essential nutrients (only catabolic cross-feeding)
    mat_ess = np.zeros((n_s,n_r))

    # no auxotrophies (anyone can produce what metabolism allows)
    spec_met = np.ones((n_s,n_r))

    # create metabolic matrix of sparcity 0.8 with non zero entries sampled from Dirichelet distribution
    met_mat = definitions.met_dir(n_r,0.8)

    # recapitulate in dictionary
    mat = {
        'uptake'  : up_mat,
        'met'     : met_mat,
        'ess'     : mat_ess,
        'spec_met': spec_met,
        'sign'    : sign_mat
    }

    # definition of the rest of the model parameters

    # growth and maintainence
    g = np.ones((n_s))
    m = np.zeros((n_s))+0.1

    if supplied > 1:
        # extraction of resources to supply
        ext_indices = np.random.sample(range(1, n_r+1), supplied)
    else:
        ext_indices = []

    # reinsertion of chemicals
    tau = np.zeros((n_r))+10
    ext = np.zeros((n_r))
    # primary carbon source replenished to saturation
    ext[0] = 10
    ext[ext_indices] = 10
    # initial guess for resources
    guess = np.ones((n_r))*10

    # define parameters
    param = {
        'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
        'l'  : np.ones((n_r))*leakage,                     # leakage               [adim]
        'g'  : g,                                          # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext,                                        # external replenishment  
        'tau' : tau,                                       # chemicals dilution                             
        'tau_s': 10,                                       # species dilution
        'guess_wm': guess                                  # initial resources guess
    }

    N_fin,R_fin=well_mixed.run_wellmixed(np.ones((n_s))*0.1,param,mat,well_mixed.dR_dt_maslov,well_mixed.dN_dt_maslov,3000)

    # calculate matrices for mapping
    grad_mat=grad_i_alpha(R_fin[-1,:],param,mat)
    sig=sigma(R_fin[-1,:],param)
    f_mat=f_i_alpha(R_fin[-1,:],param,mat)

    # intrinsic growth rates
    g_LV = np.zeros((n_s))
    for i in range(n_s):
        g_LV[i]=np.dot(grad_mat[i],sig)

    # interaction matrix
    A_int=np.zeros((n_s,n_s))
    for i in range(n_s):
        for j in range(n_s):
            A_int[i,j]=np.dot(grad_mat[i],f_mat[j])

    # Solve Lotka-Volterra dynamics
    lv_args = (g_LV,A_int,n_s)
    t_span_lv = (0,10000)
    t_eval_lv = np.arange(t_span_lv[0],t_span_lv[1],0.1)
    solLV = solve_ivp(fun=LV_model, t_span=t_span_lv, y0=np.ones((n_s))*0.1, t_eval=t_eval_lv, args=lv_args)

    # save in output file all parameters 
    with open(f"{results_dir}/parameters.txt", 'w') as file:
        file.write("Simulation Parameters:\n")
        for key, value in param.items():
            file.write(f"{key}: {value}\n")
        file.write("\nMatrices:\n")
        for key, value in mat.items():
            file.write(f"{key}:\n{value}\n\n")

        data = {
            'n_resources':n_resources,
            'supplied':supplied,
            'average_consumed':average_consumed,
            'leakage':leakage,
            'replica':replica,
            'uptake':up_mat,
            'D':met_mat,
            'CR_R':R_fin,
            'CR_N':N_fin,
            'LV': solLV.y,
            'g0':g_LV,
            'A':A_int
    }

    # Specifica il percorso del file di output
    output_file = f'{results_dir}/all_data.pkl'

    # Salva i dati nel file pickle
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 



#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Check if there are command-line arguments
    if len(sys.argv) == 6:
        n_resources = float(sys.argv[1])
        supplied = float(sys.argv[2])
        average_consumed = float(sys.argv[3])
        leakage = float(sys.argv[4])
        replica = float(sys.argv[5])

        # Run the simulation with the provided parameters
        CR_and_mapping(n_resources,supplied,average_consumed,leakage,replica)
        print(f"Simulation completed for {n_resources} resources, {supplied} supplied, {average_consumed} consumed and leakage {leakage}, replica {replica}")

    else:
        print("Usage: python CR_generation.py")