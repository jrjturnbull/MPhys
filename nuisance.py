import numpy as np
from numpy import float64
from numpy.linalg import inv
import math
import sys
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
from numpy.linalg.linalg import eig

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
eigenvalues = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)
x_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

# EXTRACTS NONZERO EIGENVALUES
nz_eigen = [i for i in range(len(eigenvalues)) if eigenvalues[i] > 1e-5]
eigenvalues_nz = np.array([eigenvalues[i] for i in nz_eigen])
eigenvectors_nz = np.array([eigenvectors[i] for i in nz_eigen])

l = len(eigenvalues_nz)

# COMPUTES THE NUISANCE PARAMETER EXPECTATION VALUES
nuisance_params = np.zeros(shape=l, dtype=float64)
CS = inv(th_covariance_matrix + exp_covariance_matrix)

for a in range(0, l):
    print("Computing NPE {0} of {1}...".format(a+1, l), end='\r')
    beta = eigenvectors_nz[a]

    mat = np.einsum('j,ij',beta,CS)
    TD = theory_data - exp_data
    nuisance_params[a] = np.einsum('i,i', mat, TD)
    
print("Computed all NPEs                                          ")
print(eigenvalues_nz)

"""
*********************************************************************************************************
UNCERTAINTIES
_________________________________________________________________________________________________________

"""

def t1(a, b):
    e = 1 if a == b else 0
    return e

def t2(a, b):
    beta_a = eigenvectors_nz[a]
    beta_b = eigenvectors_nz[b]
    
    e = 0
    for i in range(l):
        for j in range(l):
            e += beta_a[i] * CS[i,j] * beta_b[j]
    return e

def t3(a, b): # manual index sums
    beta_a = eigenvectors_nz[a]
    beta_b = eigenvectors_nz[b]

    vec1 = np.zeros(shape=l)
    for i in range(l):
        x = 0
        for j in range(l):
            x += beta_a[i] * CS[i,j]
        vec1[i] = x
    
    vec2 = np.zeros(shape=l)
    for i in range(l):
        x = 0
        for j in range(l):
            x += CS[i,j] * beta_b[j]
        vec2[i] = x
    
    e = 0
    for i in range(l):
        for j in range(l):
            e += vec1[i] * x_matrix[i,j] * beta_b[j]
    return e

# COMPUTES THE NUCLEAR, PDF AND TOTAL UNCERTAINTIES
Z = np.zeros(shape=(l,l))
Z_bar = np.zeros(shape=(l,l))
Z_pdf = np.zeros(shape=(l,l))
for a in range(l):
    print("Computing NPE uncertainty {0} of {1}...".format(a+1, l), end='\r')
    for b in range(l):
        t_1 = t1(a,b)
        t_2 = t2(a,b)
        t_3 = t3(a,b)

        Z[a,b] = t_1 - t_2
        Z_pdf[a,b] = -t_3
        Z_bar[a,b] = t_1 - t_2 - t_3
print("Computed all NPE uncertainties                                        ")

for i in range(l):
    print(Z[i,i], end='\t')
    print(Z_pdf[i,i], end='\t')
    print(Z_bar[i,i])

"""

uncertainties_nuc = np.array([math.sqrt(abs(Z[i,i])) for i in range(l)])
uncertainties_pdf = [math.sqrt(abs(Z_pdf[i,i])) for i in range(l)]
uncertainties_tot = [math.sqrt(abs(Z_bar[i,i])) for i in range(l)]


x = np.arange(len(nuisance_params))
y = nuisance_params
plt.scatter(x,y)
plt.errorbar(x,y,yerr=uncertainties_nuc, ls='none')
plt.show()


# DUMPS MATRICES TO FILE
nuisance_params.dump("matrices/NPE_" + root + ".dat")
Z.dump("matrices/Z_" + root + ".dat")
Z_pdf.dump("matrices/ZP_" + root + ".dat")
Z_bar.dump("matrices/ZN_" + root + ".dat")
uncertainties_nuc.dump("matrices/ZNE_" + root + ".dat")
uncertainties_pdf.dump("matrices/ZPE_" + root + ".dat")
uncertainties_tot.dump("matrices/ZTE_" + root + ".dat")

"""