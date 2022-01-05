"""
--------------------------------------------------------------------------
TO BE ADDED TO....
--------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.linalg import eigh
import sys
import math
from matplotlib.colors import LinearSegmentedColormap

root = sys.argv[1]

theory_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
correlation_matrix = np.load("matrices/CR_" + root + ".dat", allow_pickle=True)
nuclear_uncertainty_array = np.load("matrices/NUA_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvalues = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
experimental_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
pdf_covariance_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

# nuisance_parameters = np.load("matrices/NPE_" + root + ".dat", allow_pickle=True)
# uncertainties_nuc = np.load("matrices/ZNE_" + root + ".dat", allow_pickle=True)
# uncertainties_pdf = np.load("matrices/ZPE_" + root + ".dat", allow_pickle=True)
# uncertainties_tot = np.load("matrices/ZTE_" + root + ".dat", allow_pickle=True)

# autoprediction = np.load("matrices/AP_" + root + ".dat", allow_pickle=True)

# th_contribution_1 = np.load("matrices/TH1_" + root + ".dat", allow_pickle=True)
# th_contribution_2 = np.load("matrices/TH2_" + root + ".dat", allow_pickle=True)
# x_contribution_1 = np.load("matrices/X1_" + root + ".dat", allow_pickle=True)
# x_contribution_2 = np.load("matrices/X2_" + root + ".dat", allow_pickle=True)

n_dat_nz = np.shape(nuclear_uncertainty_array)[0]
n_nuis = np.shape(nuclear_uncertainty_array)[1]

# ATTEMPT TO REPLICATE THE COLORBAR USED IN THE LITERATURE (STILL NOT QUITE RIGHT...)
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

# HEATMAP OF EXPERIMENTAL COVARIANCE MATRIX
experimental_covariance_matrix_norm = np.zeros_like(experimental_covariance_matrix)
for i in range(len(experimental_covariance_matrix)):
    for j in range(len(experimental_covariance_matrix)):
        experimental_covariance_matrix_norm[i,j] = experimental_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(experimental_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-100, vmax=100), cmap=cmap)
plt.colorbar()
plt.title("Heatmap of experimental covariance matrix, normalised to the theory")
#plt.show()
plt.savefig("output/exp_covariance")
plt.clf()

# HEATMAP OF THEORY COVARIANCE MATRIX
theory_covariance_matrix_norm = np.zeros_like(theory_covariance_matrix)
for i in range(len(theory_covariance_matrix)):
    for j in range(len(theory_covariance_matrix)):
        theory_covariance_matrix_norm[i,j] = theory_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(theory_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-5, vmax=5), cmap=cmap)
plt.colorbar()
plt.title("Heatmap of theory covariance matrix, normalised to the theory")
#plt.show()
plt.savefig("output/th_covariance")
plt.clf()

# HEATMAP OF PDF UNCERTAINTIES
pdf_covariance_matrix_norm = np.zeros_like(pdf_covariance_matrix)
for i in range(len(pdf_covariance_matrix)):
    for j in range(len(pdf_covariance_matrix)):
        pdf_covariance_matrix_norm[i,j] = pdf_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(pdf_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-5, vmax=5), cmap=cmap)
plt.colorbar()
plt.title("Heatmap of PDF covariance matrix, normalised to the theory")
#plt.show()
plt.savefig("output/pdf_covariance")
plt.clf()

pdf_correlation_matrix = np.zeros_like(pdf_covariance_matrix)
for i in range(len(pdf_covariance_matrix)):
    for j in range(len(pdf_covariance_matrix)):
        pdf_correlation_matrix[i,j] = pdf_covariance_matrix[i,j] / math.sqrt(pdf_covariance_matrix[i,i] * pdf_covariance_matrix[j,j])
plt.imshow(pdf_correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar()
plt.title("Heatmap of PDF correlation matrix")
#plt.show()
plt.savefig("output/pdf_correlation")
plt.clf()


# PLOT OF DIAGONAL ELEMENTS NORMALISED TO THE THEORY
C_norm = np.zeros(shape = len(experimental_covariance_matrix))
l = experimental_covariance_matrix.shape[0]
for i in range(l):
    C_norm[i] = experimental_covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

S_norm = np.zeros(shape = len(theory_covariance_matrix))
l = theory_covariance_matrix.shape[0]
for i in range(l):
    S_norm[i] = theory_covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

X_norm = np.zeros(shape = len(pdf_covariance_matrix))
l = pdf_covariance_matrix.shape[0]
for i in range(l):
    X_norm[i] = pdf_covariance_matrix[i,i] / (theory_data[i] * theory_data[i])

x = np.arange(len(C_norm))
plt.scatter(x, C_norm, c='g', s=1.5)
plt.scatter(x, S_norm, c='b', s=1.5)
plt.scatter(x, X_norm, c='r', s=1.5)
plt.title("Diagonal elements of C (green), S (blue), X (red), normalised to the theory")
#plt.show()
plt.savefig("output/diagonal_elements.png")
plt.clf()

# NON-ZERO EIGENVALUES
nz_eigen = [i for i in range(len(eigenvalues)) if eigenvalues[i] > 1e-5]
eigenvalues_nz = np.array([eigenvalues[i] for i in nz_eigen])[::-1]

x = np.arange(len(eigenvalues_nz))
plt.scatter(x, eigenvalues_nz, s=1.5)
plt.title("Non-zero eigenvalues of S (cutoff = 1e-5)")
plt.yscale('log')
#plt.show()
plt.savefig("output/nz_eigenvalues.png")
plt.clf()

""" CURRENTLY BROKEN DUE TO ERRORS IN NUISANCE.PY & AUTOPREDICTION.PY

# NUISANCE PARAMETER EXPECTATION VALUES
plt.scatter(x, nuisance_parameters
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_nuc, ls='none'))
plt.title("Nuisance parameters with nuclear uncertainties")
plt.savefig("output/NPE_nuc.png)
plt.clf()

plt.scatter(x, nuisance_parameters
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_pdf, ls='none'))
plt.title("Nuisance parameters with PDF uncertainties")
plt.savefig("output/NPE_pdf.png)
plt.clf()

plt.scatter(x, nuisance_parameters
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_tot, ls='none'))
plt.title("Nuisance parameters with total uncertainties")
plt.savefig("output/NPE_pdf.png)
plt.clf()


# AUTOPREDICTION SHIFTS COMPARED TO THEORY-DATA DIFFERENCES
# TODO

# AUTOPREDICTION MATRICES
autoprediction_norm = np.zeros_like(autoprediction)
for i in range(len(autoprediction)):
    for j in range(len(autoprediction)):
        autoprediction_norm[i,j] = autoprediction[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(autoprediction_norm, norm=SymLogNorm(1e-4,vmin=-5, vmax=5), cmap=cmap)
plt.colorbar()
plt.title("Heatmap of autoprediction covariance matrix, normalised to the theory")
#plt.show()
plt.savefig("output/autoprediction_covariance")
plt.clf()

autoprediction_corr = np.zeros_like(autoprediction)
for i in range(len(autoprediction)):
    for j in range(len(autoprediction)):
            autoprediction_corr[i,j] = autoprediction[i,j] / sqrt(autoprediction[i,i] * autoprediction[j,j])
plt.imshow(autoprediction_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Heatmap of autoprediction correlation matrix")
#plt.show()
plt.savefig("output/autoprediction_correlation")
plt.clf()

autoprediction_cons = pdf_covariance_matrix + theory_covariance_matrix

unc_ap = np.array([math.sqrt(autoprediction[i,i])/ (theory_data[i] * theory_data[i]) for i in range(len(autoprediction))])
unc_pdf = np.array([math.sqrt(pdf_covariance_matrix[i,i]) / (theory_data[i] * theory_data[i]) for i in range(len(pdf_covariance_matrix))])
unc_cons = np.array([math.sqrt(autoprediction_cons[i,i]) / (theory_data[i] * theory_data[i]) for i in range(len(autoprediction_cons))])

TODO: SCATTER PLOT OF THESE THREE UNCERTAINTIES

# CONTRIBUTIONS TO THE DIAGONAL ELEMENTS OF THE CORRELATED THEORY & PDF UNCERTAINTIES
TODO

"""