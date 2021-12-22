import numpy as np
from numpy import float64
from numpy.linalg import inv
import sys
from numpy.core.fromnumeric import shape

root = sys.argv[1]

# LOADS ALL REQUIRED MATRICES
th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)

n_nuis = shape(nuclear_uncertainty_array)[1]

# COMPUTES THE NUISANCE PARAMETER EXPECTATION VALUES
nuisance_params = np.zeros(shape=n_nuis, dtype=float64)
for a in range(0, len(eigenvectors)):
    beta = nuclear_uncertainty_array[:, a]
    #beta = eigenvectors[a]
    CS = inv(th_covariance_matrix + exp_covariance_matrix)
    mat = np.matmul(beta, CS)
    TD = theory_data - exp_data
    nuisance_params[a] = np.matmul(mat, TD)

# COMPUTES THE Z MATRIX (SLOW)
Z = np.zeros(shape=(len(eigenvectors),len(eigenvectors)))
for a in range(0, len(eigenvectors)):
    print("Computing covariance element {0} of {1}...".format(a+1, len(eigenvectors)), end='\r')
    for b in range(0, len(eigenvectors)):
        CS = inv(th_covariance_matrix + exp_covariance_matrix)
        beta_a = nuclear_uncertainty_array[:, a]
        beta_b = nuclear_uncertainty_array[:, b]
        Z[a,b] = np.kron(a,b) - np.matmul(np.matmul(beta_a, CS), beta_b)
print("Computed all elements                                                ")

# DUMPS MATRICES TO FILE
nuisance_params.dump("matrices/NPE_" + root + ".dat")
Z.dump("matrices/Z_" + root + ".dat")
