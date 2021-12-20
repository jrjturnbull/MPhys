import numpy as np
from numpy import float64
from numpy.linalg import inv
import sys
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
from scipy.linalg.decomp import eig

root = sys.argv[1]

th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)

n_nuis = shape(nuclear_uncertainty_array)[1]

nuisance_params = np.zeros(shape=n_nuis, dtype=float64)

for a in range(0, len(eigenvectors)):
    beta = nuclear_uncertainty_array[:, a]
    beta = eigenvectors[a]
    CS = inv(th_covariance_matrix + exp_covariance_matrix)
    mat = np.matmul(beta, CS)
    TD = theory_data - exp_data
    nuisance_params[a] = np.matmul(mat, TD)

covariance = np.zeros(shape=(n_nuis, n_nuis) dtype=float64)
for a in range(0, len(eigenvectors)):
    for b in range(0, len(eigenvectors)):
        delta = np.kron(a, b)
        CS = inv(th_covariance_matrix + exp_covariance_matrix)
        

fix, ax = plt.subplots()
x = np.arange(len(nuisance_params))
y = nuisance_params
im = ax.scatter(x, y, marker='x', s=2)
plt.axhline(color='black', lw=0.5)
plt.show()