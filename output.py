"""
--------------------------------------------------------------------------
TO BE REWRITTEN....
--------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.linalg import eigh
import sys
import math
from matplotlib.colors import LinearSegmentedColormap

# RETURNS DIAGONAL ELEMENT DIVIDED BY CORRESPONDING EXPERIMENTAL VALUE
def compute_diagonal_element(i):
    diag = math.sqrt(covariance_matrix[i,i])
    data = exp_data[i]
    d = diag / data
    return d

root = sys.argv[1]

covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
correlation_matrix = np.load("matrices/CR_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvalues = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
experimental_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
pdf_covariance_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

n_dat_nz = np.shape(nuclear_uncertainty_array)[0]
n_nuis = np.shape(nuclear_uncertainty_array)[1]


c = ["firebrick","red","chocolate","orange","sandybrown","peachpuff","lightyellow",
        "honeydew","palegreen","aquamarine","mediumturquoise", "royalblue","midnightblue"]
v = [0,.1,.2,.3,.4,.45,.5,.55,.6,.7,.8,.9,1]
l = list(zip(v,reversed(c)))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

"""
*********************************************************************************************************
OUTPUT
_________________________________________________________________________________________________________

"""

# PLOT OF DIAGONAL ELEMENTS NORMALISED TO THE THEORY
C_norm = np.zeros(shape = len(experimental_covariance_matrix))
l = experimental_covariance_matrix.shape[0]
for i in range(l):
    C_norm[i] = experimental_covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

S_norm = np.zeros(shape = len(covariance_matrix))
l = covariance_matrix.shape[0]
for i in range(l):
    S_norm[i] = covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

X_norm = np.zeros(shape = len(pdf_covariance_matrix))
l = pdf_covariance_matrix.shape[0]
for i in range(l):
    X_norm[i] = pdf_covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

x = np.arange(len(C_norm))
#plt.scatter(x, C_norm, c='g')
plt.scatter(x, S_norm, c='b', s=1.5)
plt.scatter(x, X_norm, c='r', s=1.5)
plt.show()