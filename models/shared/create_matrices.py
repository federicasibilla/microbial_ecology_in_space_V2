"""
create_matrices.py: file to store the functions to define networks and matrices structures

CONTAINS: - up_binary: function to create a binary uptake matrix
          - met_dir:   function to sample D coefficients from Dirichelet distribution, with fixed sparcity

MATERIALS: see https://images.prismic.io/rosetta-marketing-website/722fd521-3acf-4826-8335-1db031957aa2_image+%2850%29.png?auto=compress,format
           for an intuitive graphical explaination of how Dirichlet distribution works for a parameters space with 3 dimensions

"""

import numpy as np
import sys

#-------------------------------------------------------------------------------------------------------------
# up_binary: function to create a binary uptake matrix, the user specifies how many resources each species can 
# consume, there are no tradeoffs implemented; at least one species is going to consume PCS

def up_binary(n_s,n_r,n_pref):

    """
    n_s:    int, number of species
    n_r:    int, number of resources
    n_pref: int, number of resources consumed by each species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    # check that number of consumed resources is correct
    if n_pref>n_r:
        print('number of consumed resources excedes total number of resources')
        sys.exit()

    # initialize uptake matrix empty
    up_mat = np.zeros((n_s,n_r))

    # assign randomly preferred resources
    for i in range(n_s):
        ones_indices = np.random.choice(n_r, n_pref, replace=False)
        up_mat[i, ones_indices] = 1

    # if there is no PCS consumer, add randomly a preference for PCS
    if (up_mat[:,0] == 0).all():
        up_mat[np.random.randint(0, n_s-1),0] = 1

    return up_mat

#-------------------------------------------------------------------------------------------------------------
# met_dir: function to create a D-sampled D matrix with given sparcity

def met_dir(n_r,sparcity):

    # select which entries are non-zero based on spersity
    met_mat = np.ones((n_r,n_r))*(np.random.rand(n_r, n_r) > sparcity)    
    # PCS is not produced  
    met_mat[0,:] = 0    
    # diagonal elements set to zero (no self-production)                                                    
    np.fill_diagonal(met_mat, 0)                                            
    # PCS is transformed in at least one other product
    if (met_mat[:,0] == 0).all():
        met_mat[np.random.randint(0, n_r-1),0] = 1

    # sample non-zero entries of each column from D. distribution
    for column in range(n_r):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(met_mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from symmetric Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            met_mat[non_zero_indices, column] = dirichlet_values

    return met_mat

