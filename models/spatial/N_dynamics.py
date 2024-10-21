"""
N_dynamics.py: file containing the functions related to the population grid and its update.
               This corresponds to the code that determines the cellular automaton update rules.

CONTAINS: - growth_rates: the function to compute growth rates starting from equilibrium R and
            the state vector of the population grid
          - growth_rates_partial: function with partial rgulation on uptake
          - growth_rates_maslov: same but with Maslov-like regulation on uptake 
          -------------------------------------------------------------------
          - birth_death_biomass: function to update the grid based on biomass
          -------------------------------------------------------------------
          - encode: function to one hot encode species matrix
          - decode: function to decode species matrix
          - get_neighbour: function to choose a random neigh. to substitute, given an index

"""

import numpy as np

#---------------------------------------------------------------------------------------------
# birth_death_biomass: function implementing the cellular automaton step based on biomass 
# accumulation by cells; assumes that the model has a marix where to store the biomass variable

def birth_death_biomass(N, g, param, mat, biomass):

    """
    Update the grid by finding the first cell reaching division threshold

    N: matrix, nxnxn_s, state of the population
    g: matrix, nxn, growth rates at this moment
    param: dict, parameters
    mat: dict, matrices
    biomass: matrix, nxn, current biomass values at each point on the grid

    RETURNS 
    - new_N: matrix, nxnxn_s, new state of population
    - new_biomass: matrix, nxn, updated biomass matrix
    - t_div: float, time of division event

    """

    n = N.shape[0]

    # calculate division times of all cells 
    times_mat = np.log(2/biomass)/g
    # find the fastest: first devider
    t_div = np.min(times_mat)
    i, j = np.random.choice(np.transpose(np.where(times_mat == np.min(times_mat))), size=1)[0]

    # Look at the 8 neighbors with periodic boundary conditions
    neighbors_i = np.array([(i-1) % n, i, (i+1) % n, (i+1) % n, (i+1) % n, i, (i-1) % n, (i-1) % n])
    neighbors_j = np.array([(j-1) % n, (j+1) % n, (j+1) % n, j, (j-1) % n, (j-1) % n, (j-1) % n, (j+1) % n])

    # replace a random neighbour
    nei_idx = np.random.randint(0,8)
    neighbor_i, neighbor_j = neighbors_i[nei_idx], neighbors_j[nei_idx]
    # change identity
    N[neighbor_i,neighbor_j]=N[i,j]

    # update all other biomasses
    new_biomass = biomass * np.exp(g*t_div)
    # Update biomass of mother and daughter with random noise
    noise = np.random.uniform(-0.1, 0.1)  # Generate some small noise
    new_biomass[i, j] = 1 + noise
    new_biomass[neighbor_i, neighbor_j] = 1 - noise 
    
    return N, new_biomass, t_div

#---------------------------------------------------------------------------------------------
# growth_rates(R,N,param,mat) function to calculate the growth rates of each individual
# based on their intrinsic conversion factor and on the concentrations of resources on the
# underlying grids of equilibrium concentrations ATT: there is no maintainance: no negative gr

def growth_rates(R, N, param, mat):

    """
    Calculate growth rates of each individual and modulate growth based on limiting nutrients.
    
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS:
    - growth_matrix: matrix, nxn, growth rates of each individual
    - mod:           matrix, nxnx2, first layer is mu, second layer is limiting nutrient index
    
    """
    
    n, _, n_r = R.shape
    n_s = N.shape[2]
    
    species = np.argmax(N, axis=2)
    growth_matrix = np.zeros((n, n))

    # Identify auxotrophies on the grid
    mask = (mat['ess'][species] != 0).astype(int)
    
    # Calculate Michaelis-Menten at each site and mask for essential resources
    upp = R / (R + 1)
    up_ess = np.where(mask == 0, 1, upp)
    
    # Find limiting nutrient and calculate corresponding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim = np.min(up_ess, axis=2)
    
    # Create modulation mask
    mu = np.ones_like(R) * mu_lim[:, :, np.newaxis]
    mu[np.arange(n)[:, None], np.arange(n), lim] = 1
    
    # Modulation matrix
    mod = np.zeros((n, n, 2))
    mod[:, :, 0] = mu_lim
    mod[:, :, 1] = lim
    
    # Modulate uptake and insert uptake rates
    uptake = upp * mu * mat['uptake'][species]

    for i in range(n_s):
        #realized_met = mat['met'] * mat['spec_met'][i][:, np.newaxis]
        l = param['l'].copy()
        ## leakage adjusted to zero if nothing is produced
        #l[np.sum(realized_met, axis=1) == 0] = 0

        species_i_matrix = N[:, :, i]
        growth_matrix += species_i_matrix * param['g'][i] * (
            np.sum(uptake * mat['sign'][i] * (1 - l), axis=2)
        )
        
    return growth_matrix, mod

#--------------------------------------------------------------------------------------------
# growth_rates(R,N,param) function to calculate the growth rates of each individual
# based on their intrinsic conversion factor and on the concentrations of resources on the
# underlying grids of equilibrium concentrations

def growth_rates_partial(R, N, param, mat):

    """
    Calculate growth rates of each individual and modulate growth based on limiting nutrients.
    
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS:
    - growth_matrix: matrix, nxn, growth rates of each individual
    - mod:           matrix, nxnx2, first layer is mu, second layer is limiting nutrient index
    
    """
    
    n, _, n_r = R.shape
    n_s = N.shape[2]
    
    species = np.argmax(N, axis=2)
    growth_matrix = np.zeros((n, n))

    # Identify auxotrophies on the grid
    mask = (mat['ess'][species] != 0).astype(int)
    
    # Calculate Michaelis-Menten at each site and mask for essential resources
    upp = R / (R + 1)
    up_ess = np.where(mask == 0, 1, upp)
    
    # Find limiting nutrient and calculate corresponding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim = np.min(up_ess, axis=2)
    
    # Create modulation mask
    mu = np.ones_like(R) * mu_lim[:, :, np.newaxis]
    mu[np.arange(n)[:, None], np.arange(n), lim] = 1
    
    # Modulation matrix
    mod = np.zeros((n, n, 2))
    mod[:, :, 0] = mu_lim
    mod[:, :, 1] = lim
    
    # Modulate uptake and insert uptake rates
    uptake = upp * (param['alpha'] +(1-param['alpha'])*mu) * mat['uptake'][species]

    for i in range(n_s):
        #realized_met = mat['met'] * mat['spec_met'][i][:, np.newaxis]
        l = param['l'].copy()
        ## leakage adjusted to zero if nothing is produced
        #l[np.sum(realized_met, axis=1) == 0] = 0

        species_i_matrix = N[:, :, i]
        growth_matrix += species_i_matrix * param['g'][i] * (
            np.sum(uptake * mat['sign'][i] * (1 - l), axis=2)
        )
        
    return growth_matrix, mod

#-------------------------------------------------------------------------------------------------
# growth_rates(R,N,param) function to calculate the growth rates of each individual
# assuming leakage is modulated by Liebig's law of minimum

def growth_rates_maslov(R, N, param, mat):

    """
    Calculate growth rates of each individual and modulate growth based on limiting nutrients.
    
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS:
    - growth_matrix: matrix, nxn, growth rates of each individual
    - mod:           matrix, nxnx2, first layer is mu, second layer is limiting nutrient index
    
    """
    
    n, _, n_r = R.shape
    n_s = N.shape[2]
    
    species = np.argmax(N, axis=2)
    growth_matrix = np.zeros((n, n))

    # Identify auxotrophies on the grid
    mask = (mat['ess'][species] != 0).astype(int)
    
    # Calculate Michaelis-Menten at each site and mask for essential resources
    upp = R / (R + 1)
    up_ess = np.where(mask == 0, 1, upp)
    
    # Find limiting nutrient and calculate corresponding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim = np.min(up_ess, axis=2)
    
    # Create modulation mask
    mu = np.ones_like(R) * mu_lim[:, :, np.newaxis]
    mu[np.arange(n)[:, None], np.arange(n), lim] = 1

    # Modulation matrix
    mod = np.zeros((n, n, 2))
    mod[:, :, 0] = mu_lim
    mod[:, :, 1] = lim

    # modulate leakage
    leakage = param['l']*mu + (1-mu) 
    
    # Modulate uptake and insert uptake rates
    uptake = upp * mat['uptake'][species]

    for i in range(n_s):

        species_i_matrix = N[:, :, i]
        growth_matrix += species_i_matrix * param['g'][i] * (
            np.sum(uptake * (1 - leakage) * mat['sign'][i], axis=2)
        )
        
    return growth_matrix, mod




#--------------------------------------------------------------------------------------
# define encoding(N) function to one-hot encode the species matrix

def encode(N, all_species):

    """
    N:           matrix, nxn, species (integer values, each integer corresponds to a species)
    all_species: list of all possible species

    RETURNS one_hot_N: matrix, nxnxn_s, species identity is one-hot-encoded

    """

    n_s = len(all_species)

    one_hot_N = np.zeros((*N.shape, n_s), dtype=int)

    # encoding
    for i, value in enumerate(all_species):
        one_hot_N[:, :, i] = (N == value).astype(int)

    return one_hot_N


#--------------------------------------------------------------------------------------
# define decoding(N) function to one-hot decode the species matrix

def decode(N):

    """
    N: matrix, nxnxn_s, containing one hot encoded species

    RETURNS decoded_N: matrix, nxn, decoded N matrix

    """

    # max index on axis m (one-hot dimension)
    decoded_N = np.argmax(N, axis=-1)

    return decoded_N

#-------------------------------------------------------------------------------------
# choose_random_neigh

def get_neighbor(i, j, direction):
    
    """
    i: int, row index of birth
    j: int, column index of birth
    direction: int, from 0 to 7, looser neig

    RETURNS indexes of neig.

    """

    # Ordered counterclockwise from bottom-left
    shifts = [(1, -1),  # 1: Bottom-left
              (0, -1),  # 2: Left
              (-1, -1), # 3: Top-left
              (-1, 0),  # 4: Top
              (-1, 1),  # 5: Top-right
              (0, 1),   # 6: Right
              (1, 1),   # 7: Bottom-right
              (1, 0)]   # 8: Bottom

    # Get the shift based on the direction (subtract 1 since list is 0-indexed)
    row_shift, col_shift = shifts[direction]

    # Return the new neighbor position
    return i + row_shift, j + col_shift