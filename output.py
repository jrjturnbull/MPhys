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
covariance_matrix_norm = np.load("matrices/CVN_" + root + ".dat", allow_pickle=True)
correlation_matrix = np.load("matrices/CR_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvalues = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)
eigenvalues_norm = np.load("matrices/EVLN_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
eigenvectors_norm = np.load("matrices/EVCN_" + root + ".dat", allow_pickle=True)
experimental_covariance_norm = np.load("matrices/ECVN_" + root + ".dat", allow_pickle=True)

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

# PLOT HEATMAP OF NORMALISED EXPERIMENTAL COVARIANCE MATRIX
fig, ax = plt.subplots()
im = ax.imshow(covariance_matrix, cmap=cmap, norm=SymLogNorm(1e-5))
plt.title("Normalised experimental covariance matrix for\n" + root)
plt.colorbar(im)
plt.savefig("output/experimental_covariance_norm_heatmap_" + root + ".png")

# PLOT HEATMAP OF NORMALISED THEORETICAL COVARIANCE MATRIX
fig.clear(True)
fig, ax = plt.subplots()
im = ax.imshow(covariance_matrix_norm, cmap=cmap, norm=SymLogNorm(1e-5))
plt.title("Normalised theoretical covariance matrix for\n" + root)
plt.colorbar(im)
plt.savefig("output/theoretical_covariance_norm_heatmap_" + root + ".png")

# PLOT HEATMAP OF CORRELATION MATRIX
fig.clear(True)
fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
plt.title("Correlation matrix for\n" + root)
plt.colorbar(im)
plt.savefig("output/correlation_matrix_heatmap_" + root + ".png")

# PLOT GRAPH OF SCALED DIAGONAL ELEMENTS
fig.clear(True)
fix, ax = plt.subplots()
x = np.arange(n_dat_nz)
y = np.zeros_like(x, dtype=np.float64)
for i in range(0, n_dat_nz):
    y[i] = compute_diagonal_element(i)
im = ax.scatter(x, y, marker='x', s=2)
plt.title("Scaled diagonal elements for\n" + root)
plt.savefig("output/diagonal_elements_" + root + ".png")

# OUTPUT NONZERO EIGENVALUES (CUTOFF = 1e-6)
eigen_plot_path = "output/eigenvalues_plot_" + root + ".png"
fig.clear(True)
fix, ax = plt.subplots()
x = np.arange(len(eigenvalues))
y = sorted(eigenvalues, reverse=True)
ax.set_yscale('log')
im = ax.scatter(x, y, marker='x')
plt.title("Eigenvalues for " + root + "\n(cutoff = 1e-6)")
plt.savefig(eigen_plot_path)

eigen_plot_path_norm = "output/eigenvalues_norm_plot_" + root + ".png"
fig.clear(True)
fix, ax = plt.subplots()
x = np.arange(len(eigenvalues_norm))
y = sorted(eigenvalues_norm, reverse=True)
ax.set_yscale('log')
im = ax.scatter(x, y, marker='x')
plt.title("Eigenvalues (normalised) for " + root + "\n(cutoff = 1e-6)")
plt.savefig(eigen_plot_path_norm)