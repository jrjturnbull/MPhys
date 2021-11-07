"""
*********************************************************************************************************
covariance.py

The main python script for this project:
    -   Reads in and interprets DATA, SYSTYPE and THEORY files for supplied root
    -   Computes nuclear covariance & correlation matrices for given DATA file (ignoring zero rows)
    -   Outputs heatmaps and diagonal element plots of the covariance & correlation matrices
    -   Normalises the covariance matrix to the theoretical data
    -   Outputs nonzero eigenvalues of the normalised covariance matrix (both raw data and plot)
    -   Saves all matrices to file
_________________________________________________________________________________________________________

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math
from numpy import float64
from scipy.linalg import eigh
import os.path
import sys

"""
*********************************************************************************************************
METHODS
_________________________________________________________________________________________________________

"""

# COMPUTES THE GIVEN COVARIANCE MATRIX ELEMENT
def compute_covariance_element(i, j, n_nuis):
    e = 0
    for n in range(0, n_nuis):
        delta_i = nuclear_uncertainty_array[i][n]
        delta_j = nuclear_uncertainty_array[j][n]
        e += (delta_i * delta_j)
    e /= n_nuis # normalisation
    return e

# COMPUTES THE GIVEN CORRELATION MATRIX ELEMENT
def compute_correlation_element(i, j):
    norm = math.sqrt(covariance_matrix[i,i] * covariance_matrix[j,j])
    return (covariance_matrix[i,j] / norm)

# RETURNS SPARSITY PERCENTAGE OF SUPPLIED ARRAY (100% = FULLY SPARSE)
def compute_sparsity(array):
    sparse_elements = 0
    for i in array:
        for j in i:
            if (j == 0): sparse_elements += 1
    sparsity = sparse_elements / array.size
    return sparsity

# RETURNS NONZERO EIGENVALUES OF SUPPLIED ARRAY (DEFAULT CUTOFF = 1e-8)
def compute_nonzero_eigenvalues(array, cutoff = 1e-8):
    w, v = eigh(array)
    nonzero = w[w > cutoff]
    return nonzero

# RETURNS DIAGONAL ELEMENT DIVIDED BY CORRESPONDING EXPERIMENTAL VALUE
def compute_diagonal_element(i):
    diag = math.sqrt(covariance_matrix[i,i])
    data = experimental_data[i]
    d = diag / data
    return d

"""
*********************************************************************************************************
MAIN PROGRAM
_________________________________________________________________________________________________________

"""

# PATH TO DATA FILES, OPTIONALLY PASSED BY ARGUMENT
tried_arg = False
while True:
    if len(sys.argv) > 1 and tried_arg == False: # first tries any supplied argument
        root = sys.argv[1]
        tried_arg = True
    else:
        root = input("Please enter the datafile root: ")
    path_data = "datafiles/DATA/DATA_" + root + ".dat"
    path_syst = "datafiles/SYSTYPE/SYSTYPE_" + root + "_DEFAULT.dat"
    path_theo = "datafiles/THEORY/THEORY_" + root + ".dat"

    if not os.path.exists(path_data):
        print("ERROR: " + path_data + " does not exist!")
        continue
    elif not os.path.exists(path_syst):
        print("ERROR: " + path_syst + " does not exist!")
        continue
    elif not os.path.exists(path_theo):
        print("ERROR: " + path_theo + " does not exist!")
        continue
    else:
        break
print()
print("Running covariance.py for: " + root)

# FIND LINES WHICH INCLUDE DATAPOINTS (INCL. ZERO) FROM SYSTYPE FILE
row_start = 1 # ignore first line
row_end = 0
with open(path_data) as file:
    linecount = len(file.readlines())
    row_end = linecount
n_dat = row_end - row_start

# FIND WHICH UNCERTANTIES ARE NUCLEAR UNCERTAINTIES (INCL. ZERO) FROM SYSTYPE FILE
nuclear_start = 0
nuclear_end = 0
with open(path_syst) as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        entries = lines[i].split("    ")
        if ("NUCLEAR" in entries[2]):
            nuclear_start = int(entries[0]) - 1
            break
    nuclear_end = len(lines) - 1 # assumes that final uncertainty is always nuclear
n_nuis = nuclear_end - nuclear_start

# READ DATA INTO NUCLEAR_UNCERTAINTY_ARRAY & EXPERIMENTAL_DATA & THEORY_VALUES, IGNORING ROWS OF ZEROS
nuclear_uncertainty_array = np.zeros(shape=(n_dat, n_nuis), dtype=float64)
experimental_data = np.zeros(shape = n_dat, dtype=float64)
theory_values = np.zeros(shape = n_dat, dtype=float64)
zero_lines = 0 # records how many zero lines have been found
with open(path_data) as data:
    with open(path_theo) as theory:
        data_lines = data.readlines()
        theory_lines = theory.readlines()
        for i in range(row_start, row_end):
            nuclear_uncertainties = data_lines[i].split("\t")[7:-1:2] # remove exp. values and mult. uncertanties
            nuclear_uncertainties = nuclear_uncertainties[nuclear_start:] # remove non-nuclear uncertainties
            nuclear_uncertainties = [float64(i) for i in nuclear_uncertainties] # convert from str to f64

            # Remove rows containing all zeros
            if(all(v == 0 for v in nuclear_uncertainties)):
                zero_lines += 1 # records that a zero line has been found
                nuclear_uncertainty_array = nuclear_uncertainty_array[:-1, :] # removes final row
                experimental_data = experimental_data[:-1] # removes final row
                theory_values = theory_values[:-1]
                continue
            nuclear_uncertainty_array[(i-1) - zero_lines] = nuclear_uncertainties
            experimental_data[(i-1) - zero_lines] = data_lines[i].split("\t")[5] # extracts data_value
            theory_values[(i-1) - zero_lines] = theory_lines[i-1]

n_dat_nz = len(nuclear_uncertainty_array) # number of non-zero data points

# DETERMINE THE COVARIANCE MATRIX
covariance_matrix = np.zeros(shape = (n_dat_nz, n_dat_nz))
for i in range(0, n_dat_nz):
    for j in range(0, n_dat_nz):
        print("Computing covariance element {0} of {1}...".format(i*n_dat_nz + j + 1, n_dat_nz*n_dat_nz), end='\r')
        covariance_matrix[i, j] = compute_covariance_element(i, j, n_nuis)
print("Computed all {0} covariance elements                                      ".format(n_dat_nz*n_dat_nz))
print("Sparsity of covariance matrix: " + "{:.2%}".format(compute_sparsity(covariance_matrix)))

# DETERMINE THE CORRELATION MATRIX
correlation_matrix = np.zeros_like(covariance_matrix)
for i in range(0, n_dat_nz):
    for j in range(0, n_dat_nz):
        print("Computing correlation element {0} of {1}...".format(i*n_dat_nz + j + 1, n_dat_nz*n_dat_nz), end='\r')
        correlation_matrix[i, j] = compute_correlation_element(i, j)
print("Computed all {0} correlation elements                            ".format(n_dat_nz*n_dat_nz))
print("Sparsity of correlation matrix: " + "{:.2%}".format(compute_sparsity(correlation_matrix)))

# NORMALISE THE COVARIANCE MATRIX
covariance_matrix_norm = np.zeros_like(covariance_matrix)
for i in range(0, n_dat_nz):
    for j in range(0, n_dat_nz):
        print("Normalising covariance element {0} of {1}...".format(i*n_dat_nz + j + 1, n_dat_nz*n_dat_nz), end='\r')
        covariance_matrix_norm[i, j] = covariance_matrix[i, j] / (theory_values[i] * theory_values[j])
print("Normalised all {0} covariance elements                            ".format(n_dat_nz*n_dat_nz))


"""
*********************************************************************************************************
OUTPUT
_________________________________________________________________________________________________________

"""

# PLOT HEATMAP OF COVARIANCE MATRIX
fig, ax = plt.subplots()
im = ax.imshow(covariance_matrix, cmap='jet', norm=LogNorm())
plt.title("Covariance matrix for\n" + root)
plt.colorbar(im)
plt.savefig("output/covariance_matrix_heatmap_" + root + ".png")

# PLOT HEATMAP OF CORRELATION MATRIX
fig.clear(True)
fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix, cmap='jet', vmin=-1, vmax=1)
plt.title("Correlation matrix for\n" + root)
plt.colorbar(im)
plt.savefig("output/correlation_matrix_heatmap_" + root + ".png")

# PLOT GRAPH OF SCALED DIAGONAL ELEMENTS
fig.clear(True)
fix, ax = plt.subplots()
x = np.arange(n_dat_nz)
y = np.zeros_like(x, dtype=float64)
for i in range(0, n_dat_nz):
    y[i] = compute_diagonal_element(i)
im = ax.scatter(x, y, marker='x', s=2)
plt.title("Scaled diagonal elements for\n" + root)
plt.savefig("output/diagonal_elements_" + root + ".png")

# OUTPUT NONZERO EIGENVALUES (CUTOFF = 1e-4)
eigen_data_path = "output/eigenvalues_data_" + root + ".dat"
eigen_plot_path = "output/eigenvalues_plot_" + root + ".png"
print("Computing normalised covariance matrix eigenvalues")
eigenvalues_cov = compute_nonzero_eigenvalues(covariance_matrix_norm, cutoff=1e-6)
with open(eigen_data_path, 'w') as eigen:
    eigen.write("Non-zero covariance eigenvalues for {0} (cutoff=1e-6)\n".format(root))
    for e in eigenvalues_cov:
        eigen.write("{:e}".format(e))
        eigen.write("\n")
fig.clear(True)
fix, ax = plt.subplots()
x = np.arange(len(eigenvalues_cov))
y = sorted(eigenvalues_cov, reverse=True)
ax.set_yscale('log')
im = ax.scatter(x, y, marker='x')
plt.title("Eigenvalues for " + root + "\n(cutoff = 1e-6)")
plt.savefig(eigen_plot_path)

# SAVE ALL COMPUTED MATRICES
nuclear_uncertainty_array.dump("matrices/NUA_" + root + ".dat")
covariance_matrix.dump("matrices/CV_" + root + ".dat")
covariance_matrix_norm.dump("matrices/CVN_" + root + ".dat")
correlation_matrix.dump("matrices/CR_" + root + ".dat")
experimental_data.dump("matrices/EXP_" + root + ".dat")
theory_values.dump("matrices/TH_" + root +".dat")

print()
