"""
N_dynamics.py: file containing the functions related to the population grid and its update.
               This corresponds to the code that determins the cellular automaton update rules.

CONTAINS: - growth_rates: the function to compute growth rates starting from equilibrium R and
            the state vector of the population grid
          - growth_rates_partial: function with partial rgulation on uptake
          - growth_rates_maslov: same but with Maslov-like regulation on uptake 
          - death_birth_periodic: update rule for one automaton step in PBC case
          - death_birth_periodic: update rule for one automaton step in DBC case
          - encode: function to one hot encode species matrix
          - decode: function to decode species matrix

"""

import numpy as np

from numpy import random

#--------------------------------------------------------------------------------------------
# growth_rates(R,N,param) function to calculate the growth rates of each individual
# based on their intrinsic conversion factor and on the concentrations of resources on the
# underlying grids of equilibrium concentrations

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



#-------------------------------------------------------------------------------------------------
# define death_birth(state,G) the rule of update of a single step of the automaton in the PBC case

def death_birth_periodic(state, G):

    """
    Perform death and birth process on the grid.

    state: matrix, nxn, containing the species grid (decoded)
    G:     matrix, nxn, containing growth rates at each site

    RETURNS:
    - state:    matrix, nxn, updated grid (decoded)
    - ancora:   string, 'vittoria' if one species has won, 'ancora' if there are more than 1 species present
    - max_spec: int, identity of the most present species

    """

    n = state.shape[0]
    
    # Choose cell to kill
    i, j = random.randint(0, n), random.randint(0, n)

    # Look at the 8 neighbors with periodic boundary conditions
    neighbors_i = np.array([(i-1) % n, i, (i+1) % n, (i+1) % n, (i+1) % n, i, (i-1) % n, (i-1) % n])
    neighbors_j = np.array([(j-1) % n, (j+1) % n, (j+1) % n, j, (j-1) % n, (j-1) % n, (j-1) % n, (j+1) % n])

    # Ensure the chosen cell is not surrounded by identical neighbors
    while np.all(state[i, j] == state[neighbors_i, neighbors_j]):
        if np.all(state[0, 0] == state):
            print('One species has taken up all colony space')
            return state, 'vittoria', state[0, 0]
        
        i, j = random.randint(0, n), random.randint(0, n)
        neighbors_i = np.array([(i-1) % n, i, (i+1) % n, (i+1) % n, (i+1) % n, i, (i-1) % n, (i-1) % n])
        neighbors_j = np.array([(j-1) % n, (j+1) % n, (j+1) % n, j, (j-1) % n, (j-1) % n, (j-1) % n, (j+1) % n])

    # Create probability vector from growth rates vector
    growth_rates = np.array([
        G[neighbors_i[k], neighbors_j[k]]
        if G[neighbors_i[k], neighbors_j[k]] >= 0 else 0
        for k in range(8)
    ])

    # Normalize growth rates to probabilities
    total_growth = np.sum(growth_rates)
    if total_growth > 0:
        probabilities = growth_rates / total_growth
    else:
        probabilities = np.ones(8) / 8

    # Choose the winner cell index based on probabilities
    winner_idx = np.random.choice(8, p=probabilities)

    # Reproduction
    state[i, j] = state[neighbors_i[winner_idx], neighbors_j[winner_idx]]

    # Calculate the most present species
    max_spec = np.argmax(np.bincount(state.ravel()))

    return state, 'ancora', max_spec


#-------------------------------------------------------------------------------------------------
# define death_sync(state,G) the rule of update of a single step of the automaton in the PBC case, synchronous update

def death_birth_sync(state, G):

    """
    Perform death and birth process on the grid.

    state: matrix, nxn, containing the species grid (decoded)
    G:     matrix, nxn, containing growth rates at each site

    RETURNS:
    - state:    matrix, nxn, updated grid (decoded)
    - ancora:   string, 'vittoria' if one species has won, 'ancora' if there are more than 1 species present
    - max_spec: int, identity of the most present species

    """
    
    n = state.shape[0]

    # Generate neighbors with periodic boundary conditions
    neighbors_i = np.array([(np.roll(state, shift, axis=0)) for shift in [-1, 1]])
    neighbors_j = np.array([(np.roll(state, shift, axis=1)) for shift in [-1, 1]])
    
    # Combine i and j shifts to get 8 neighbors
    neighbors = np.stack([neighbors_i[0], neighbors_j[1], neighbors_i[1], neighbors_j[0],
                          np.roll(state, (-1, -1), axis=(0, 1)), np.roll(state, (-1, 1), axis=(0, 1)),
                          np.roll(state, (1, -1), axis=(0, 1)), np.roll(state, (1, 1), axis=(0, 1))])

    # Calculate growth rates of all neighbors
    growth_rates_neighbors = np.array([
        np.roll(G, shift, axis) for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1),
                                                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
    ])

    # Compute probabilities from growth rates
    growth_rates = np.clip(growth_rates_neighbors, 0, None)  # Ensure non-negative growth rates
    total_growth = np.sum(growth_rates, axis=0, keepdims=True)
    probabilities = np.divide(growth_rates, total_growth, where=total_growth != 0)

    # Choose neighbor based on probabilities
    chosen_neighbors = np.array([
        np.random.choice(8, p=probabilities[:, i, j]) for i in range(n) for j in range(n)
    ]).reshape(n, n)

    # Update the state based on the chosen neighbors
    updated_state = np.choose(chosen_neighbors, neighbors)

    # Check if one species dominates
    max_spec = np.argmax(np.bincount(updated_state.ravel()))
    if np.all(updated_state == max_spec):
        return updated_state, 'vittoria', max_spec

    return updated_state, 'ancora', max_spec


#-----------------------------------------------------------------------------------------------
# define death_birth(state,G) the rule of update of a single step of the automaton in DBC case

def death_birth(state, G):
    """
    Perform death and birth process on the grid.

    state: matrix, nxn, containing the species grid (decoded)
    G:     matrix, nxn, containing growth rates at each site

    RETURNS:
    - state:    matrix, nxn, updated grid (decoded)
    - ancora:   string, 'vittoria' if one species has won, 'ancora' if there are more than 1 species present
    - max_spec: int, identity of the most present species
    """
    
    n = state.shape[0]

    # Create padded grids of states and growth rates
    padded_state = np.pad(state, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    padded_growth = np.pad(G, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)

    # Create neighborhood kernel
    kernel = np.array([[1, 1, 1],
                       [1, np.nan, 1],
                       [1, 1, 1]])

    # Choose cell to kill
    i, j = random.randint(1, n+1), random.randint(1, n+1)

    # Look at the neighbors (consider only valid neighbors)
    flat_neig = padded_state[i-1:i+2, j-1:j+2] * kernel
    flat_neig = flat_neig[~np.isnan(flat_neig)]
    
    # Only kill cells close to an interface (keep searching)
    while np.all(state[i-1, j-1] == flat_neig):

        if np.all(state[0, 0] == state):
            print('One species has taken up all colony space')
            return state, 'vittoria', state[0, 0]

        i, j = random.randint(1, n+1), random.randint(1, n+1)
        flat_neig = padded_state[i-1:i+2, j-1:j+2] * kernel
        flat_neig = flat_neig[~np.isnan(flat_neig)]

    # Create probability vector from growth rates vector
    flat_gr = padded_growth[i-1:i+2, j-1:j+2] * kernel
    flat_gr = flat_gr[~np.isnan(flat_gr)]
    flat_gr[flat_gr < 0] = 0

    if np.sum(flat_gr) != 0:
        prob = flat_gr / np.sum(flat_gr)
    else:
        prob = np.ones_like(flat_gr) / len(flat_gr)

    # Choose the winner cell index based on probabilities
    winner_idx = np.random.choice(len(flat_gr), p=prob)
    winner_id = flat_neig[winner_idx]

    # Reproduction
    state[i-1, j-1] = winner_id

    # Calculate the most present species
    max_spec = np.argmax(np.bincount(state.ravel().astype(int)))

    return state, 'ancora', max_spec


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