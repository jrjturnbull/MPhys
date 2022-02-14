"""
*********************************************************************************************************
covariance.py

Computes theory correlation/covariance matrices + eigenstuff for the supplied root:
    -   Reads in and interprets DATA, SYSTYPE and THEORY files for supplied root
    -   Computes nuclear covariance & correlation matrices for given DATA file (ignoring zero rows)
    -   Outputs eigenvalues & eigenvectors of the covariance matrix
    -   Saves all matrices to file
_________________________________________________________________________________________________________

"""
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


"""
# COMPUTES THE GIVEN COVARIANCE MATRIX ELEMENT     ---     NO LONGER USED!
def compute_covariance_element(i, j, n_nuis):
    e = 0
    for n in range(0, n_nuis):
        delta_i = nuclear_uncertainty_array[i][n]
        delta_j = nuclear_uncertainty_array[j][n]
        e += (delta_i * delta_j)
    e /= n_nuis # normalisation
    return e
"""

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
    path_cent = "datafiles/THEORY/CENT_" + root + ".dat"

    if not os.path.exists(path_data):
        print("ERROR: " + path_data + " does not exist!")
        continue
    elif not os.path.exists(path_syst):
        print("ERROR: " + path_syst + " does not exist!")
        continue
    elif not os.path.exists(path_theo):
        print("ERROR: " + path_theo + " does not exist!")
        continue
    elif not os.path.exists(path_cent):
        print("ERROR: " + path_cent + " does not exist!")
        continue
    else:
        break
print()
print("Running covariance.py for: " + root)

# READ IN C AND CS FROM EXPCOV
C = np.load("matrices/ECV_CombinedData_dw.dat", allow_pickle=True)
cs_path = "ExpCov/CS/output/tables/groups_covmat.csv"
n_dat = sum(1 for line in open(cs_path)) - 4

# READ DATA INTO NUCLEAR_UNCERTAINTY_ARRAY & EXPERIMENTAL_DATA & THEORY_VALUES, IGNORING ROWS OF ZEROS
theory_values = np.zeros(shape = n_dat, dtype=float64)
with open(path_theo) as theory:
    theory_values = theory.readlines()
theory_values = np.array([float(t) for t in theory_values])
cent_values = np.zeros(shape = n_dat, dtype=float64)
with open(path_cent) as cent:
    cent_values = cent.readlines()
cent_values = np.array([float(t) for t in cent_values])

# DETERMINE THE COVARIANCE MATRIX
""" OLD(ER) METHOD
covariance_matrix = np.zeros(shape = (n_dat_nz, n_dat_nz))
for i in range(0, n_dat_nz):
    for j in range(0, n_dat_nz):
        print("Computing covariance element {0} of {1}...".format(i*n_dat_nz + j + 1, n_dat_nz*n_dat_nz), end='\r')
        covariance_matrix[i, j] = compute_covariance_element(i, j, n_nuis)

print("Computed all {0} covariance elements                                      ".format(n_dat_nz*n_dat_nz))
#print("Sparsity of covariance matrix: " + "{:.2%}".format(compute_sparsity(covariance_matrix)))
"""

#NOTE: OLD CODE
#covariance_matrix = np.zeros(shape = (n_dat_nz, n_dat_nz))
#for n in range(n_nuis):
#    print("Computing covariance matrix term {0} of {1}...".format(n+1, n_nuis), end='\r')
#    beta = nuclear_uncertainty_array[:,n]
#    covariance_matrix += np.einsum('i,j->ij', beta, beta) #/ n_nuis
#print("Computed all {0} covariance matrix terms                            ".format(n_nuis))

#NOTE: NEW CODE
CS_valiphys = np.zeros(shape=(n_dat, n_dat))
with open(cs_path) as cs:
    lines = cs.readlines()[4:]
    for i in range(len(lines)):
        CS_valiphys[i] = lines[i].split("\t", )[3:]
covariance_matrix = CS_valiphys - C


# DETERMINE THE CORRELATION MATRIX     ---     SLIGHTLY SLOW BUT STILL WORKS
correlation_matrix = np.zeros_like(covariance_matrix)
for i in range(0, n_dat):
    for j in range(0, n_dat):
        #print("Computing correlation element {0} of {1}...".format(i*n_dat_nz + j + 1, n_dat_nz*n_dat_nz), end='\r')
        correlation_matrix[i, j] = compute_correlation_element(i, j)
#print("Computed all {0} correlation elements                            ".format(n_dat_nz*n_dat_nz))
#print("Sparsity of correlation matrix: " + "{:.2%}".format(compute_sparsity(correlation_matrix)))

# EIGENSTUFF
w, v = eigh(covariance_matrix)
#nz_eigen = [i for i in range(len(w)) if w[i] > 1e-3]       non-zero filter moved to nuisance.py
eigenvalues_cov = w #[w[i] for i in nz_eigen]
eigenvectors_cov = v #[v[i] for i in nz_eigen]
eval = np.array(eigenvalues_cov)
evec = np.array(eigenvectors_cov)
#idx = eval.argsort()[::-1]
#eval = eval[idx]
#evec = evec[:,idx]

# NORMALISED EIGENSTUFF
covariance_matrix_norm = np.zeros_like(covariance_matrix)
for i in range(len(covariance_matrix)):
    for j in range(len(covariance_matrix)):
        covariance_matrix_norm[i,j] = covariance_matrix[i,j] / (theory_values[i] * theory_values[j])
eval_norm, evec_norm = eigh(covariance_matrix_norm)
eval_norm = np.array(eval_norm)
evec_norm = np.array(evec_norm)
#idx = eval_norm.argsort()[::-1]
#eval_norm = eval_norm[idx]
#evec_norm = evec_norm[:,idx]

"""
*********************************************************************************************************
OUTPUT
_________________________________________________________________________________________________________

"""

# SAVE ALL COMPUTED MATRICES
covariance_matrix.dump("matrices/CV_" + root + ".dat")
correlation_matrix.dump("matrices/CR_" + root + ".dat")
theory_values.dump("matrices/TH_" + root + ".dat")
cent_values.dump("matrices/EXP_" + root + ".dat")

eval.dump("matrices/EVL_" + root + ".dat")
evec.dump("matrices/EVC_" + root + ".dat")

eval_norm.dump("matrices/EVLN_" + root + ".dat")
evec_norm.dump("matrices/EVCN_" + root + ".dat")

print()
