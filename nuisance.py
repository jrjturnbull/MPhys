import numpy as np
from numpy import float64
from numpy.linalg import inv
import sys

from numpy.core.fromnumeric import shape

root = sys.argv[1]

th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)

n_nuis = shape(nuclear_uncertainty_array)[1]

nuisance_params = np.zeros(shape=n_nuis, dtype=float64)

for alpha in range(0, len(eigenvectors)):
    beta = nuclear_uncertainty_array[:, alpha]
    beta = eigenvectors[alpha]
    CS = inv(th_covariance_matrix)
    mat = np.matmul(beta, CS)
    TD = theory_data - exp_data
    nuisance_params[alpha] = np.matmul(mat, TD)

print(nuisance_params)