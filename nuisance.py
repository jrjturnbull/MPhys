import numpy as np
from numpy import float64
from numpy.linalg import inv
import math
import sys
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt

print()

if (len(sys.argv) == 2):
    root = sys.argv[1]
else:
    root = "CombinedData_dw"

# LOADS ALL REQUIRED MATRICES
th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
x_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

n_nuis = shape(nuclear_uncertainty_array)[1]

# COMPUTES THE NUISANCE PARAMETER EXPECTATION VALUES
print("Computing nuisance parameter expectation values (NPEs)")
nuisance_params = np.zeros(shape=n_nuis, dtype=float64)
CS = inv(th_covariance_matrix + exp_covariance_matrix)
for a in range(0, len(eigenvectors)):
    beta = nuclear_uncertainty_array[:, a]
    mat = np.matmul(beta, CS)
    TD = theory_data - exp_data
    nuisance_params[a] = np.matmul(mat, TD)

# COMPUTES THE NUCLEAR, PDF AND TOTAL UNCERTAINTIES
print("Computing NPE uncertainties...")
Z = np.zeros(shape=(len(eigenvectors),len(eigenvectors)))
Z_bar = np.zeros(shape=(len(eigenvectors),len(eigenvectors)))
Z_pdf = np.zeros(shape=(len(eigenvectors),len(eigenvectors)))
for a in range(0, len(eigenvectors)):
    print("Computing Z-matrix row {0} of {1}...".format(a+1, len(eigenvectors)), end='\r')
    for b in range(0, len(eigenvectors)):
        beta_a = nuclear_uncertainty_array[:, a]
        beta_b = nuclear_uncertainty_array[:, b]
        t_1 = 1 if a == b else 0
        t_2 = np.matmul(beta_a, np.matmul(CS, beta_b))
        t_3 = np.matmul(beta_a, np.matmul(CS, np.matmul(x_matrix, np.matmul(CS, beta_b))))
        Z[a,b] = t_1 - t_2
        Z_pdf[a,b] = -t_3
        Z_bar[a,b] = t_1 - t_2 - t_3
print("Computed all Z-matrix rows                                        ")
uncertainties_nuc = [math.sqrt(Z[i,i]) for i in range(len(eigenvectors))]
uncertainties_pdf = [math.sqrt(Z_pdf[i,i]) for i in range(len(eigenvectors))]
uncertainties_tot = [math.sqrt(Z_bar[i,i]) for i in range(len(eigenvectors))]

# DUMPS MATRICES TO FILE
nuisance_params.dump("matrices/NPE_" + root + ".dat")
Z.dump("matrices/Z_" + root + ".dat")
Z_pdf.dump("matrices/ZP_" + root + ".dat")
Z_bar.dump("matrices/ZN_" + root + ".dat")
uncertainties_nuc.dump("matrices/ZNE_" + root + ".dat")
uncertainties_pdf.dump("matrices/ZPE_" + root + ".dat")
uncertainties_tot.dump("matrices/ZTE_" + root + ".dat")
