"""
competition_facilitation_CR.py: file to generate CR networks with different levels of competition and facilitation
                                and classify them, reference: file:///Users/federicasibilla/Downloads/journal.pcbi.1009584.pdf

CONTAINS: - generate_C: function to generate the uptake matrix, once specified how competitive we want it
          - generate_D: function to generate the metabolic matrix, once specified how facilitating we want it

          - calculate_C: function to quantify competition in a network

"""

import sys
import os
import pickle
import random

import numpy as np

path = os.path.splitext(os.path.abspath(__file__))[0]
base_path = path.split('/microbial_ecology_in_space')[0]
module_path = f'{base_path}/microbial_ecology_in_space/models/shared'

# Add the directory to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

from definitions import *

#------------------------------------------------------------------------------------------------------------------------------
# generate_C: function to generate the uptake matrix (assumes binary uptake matrices)

def generate_C(n_s,n_r,kc,beta):

    """
    n_s:    int, number of species
    n_r:    int, number of resources
    kc:     float, competition parameter between 0 and 1
    beta:   int, expected average number of resources consumed by one species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    # initialize empty C matrix
    C = np.zeros((n_s,n_r))

    # probability vector starts off as uniform (assuming same quality for all)
    p_alpha = np.ones((n_r))/n_r

    # iterate on number of species: extract consumption vectors in succession
    for i in range(n_s):

        # first extract the number of resources the species uses
        # from an exponential distr. with rate parameter beta
        # initialize m_alpha
        m_alpha = n_r+1
        while m_alpha > n_r:
            m_alpha = int(np.random.exponential(scale=beta))

        if m_alpha!=0:
            # sample m_alpha resource with probability vector p_alpha
            indexes = np.random.choice(np.arange(n_r), size=m_alpha, replace=False, p=p_alpha)
            C[i][indexes]=1

        # adjust probability vector
        # cumulative demands for each resource j at a given  i iteration dij (vector long n_r)
        d_ij = np.sum(C[:i], axis=0)
        if np.sum(d_ij)!=0:
            p_alpha = (1-kc)/n_r+kc*d_ij/np.sum(d_ij)

    return C

#------------------------------------------------------------------------------------------------------------------------------
# generate_D: function to generate the metabolic matrix 

def generate_D(n_r, s, kf, C):

    """
    n_r: int, number of resources
    s:   float, sparcity, goes from 0 (fully connected) to 1 (1 to 1)
    kf:  float, facilitiation parameter from 0 (unstructures) to 1 (fully structured)
    C:   matrix, n_sxn_r, uptake matrix

    RETURNS: D, matrix, n_rxn_r metabolic matrix
    """

    # initialize empty D
    D = np.zeros((n_r,n_r))

    # total demand for each resource
    d = np.sum(C, axis=0)

    # parameters for Dirichelet distribution for one row of D
    # D. distribution ensures they sum to 1
    for j in range(n_r):
        
        q_jk = 1/s*(1+kf*d)
        D[j] = np.random.dirichlet(q_jk)

    return D

#------------------------------------------------------------------------------------------------------------------------------
# calculate_C: function to quantify competition in a network

def calculate_C(C_matrix):

    """
    C_matrix: matrix, n_sxn_r, uptake matrix

    RETURNS C: float, value quantifying competition in a network
    """

    return np.mean(np.dot(C_matrix,C_matrix.T))

#------------------------------------------------------------------------------------------------------------------------------
# calculate_C: function to quantify competition in a network

def calculate_F(D_matrix,C_matrix):

    """
    C_matrix: matrix, n_sxn_r, uptake matrix

    RETURNS C: float, value quantifying competition in a network
    """


    return np.mean(np.dot(C_matrix,np.dot(D_matrix,C_matrix.T)))