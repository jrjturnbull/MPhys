import numpy as np
from numpy import float64
from numpy.linalg import inv, norm
import math
import sys

print()

if (len(sys.argv) == 2):
    root = sys.argv[1]
else:
    root = "CombinedData_dw" # FOR EASE OF TESTING

# LOADS ALL REQUIRED MATRICES
th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVCN_" + root + ".dat", allow_pickle=True)
eigenvalues = np.load("matrices/EVLN_" + root + ".dat", allow_pickle=True)
x_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

for i in range(len(eigenvectors)):
    eigenvectors[i] *= eigenvalues[i]

## NUA MATRIX
#datafile = open("datafiles/DATA/DATA_CombinedData_dw.dat")
#cuts = [int(c) for c in open("datafiles/CUTS/CUTS_CombinedData_dw.dat").readlines()]
#nua = np.zeros(shape=(993,100))
#lines = datafile.readlines()[1:]
#cut=0
#for i in range(len(lines)):
#    if(i in cuts):
#        cut+=1
#        continue
#    nua[i-cut:] = lines[i].split('\t')[7:-1:2]

# COMPUTES THE NUISANCE PARAMETER EXPECTATION VALUES
TD = theory_data - exp_data

## NORMALISATION
#ll = len(theory_data)
#for i in range(ll):
#    TD[i] /= theory_data[i]
#    nua[i,:] /= theory_data[i]
#    for j in range(ll):
#        th_covariance_matrix[i,j] /= (theory_data[i] * theory_data[j])
#        exp_covariance_matrix[i,j] /= (theory_data[i] * theory_data[j])
#        x_matrix[i,j] /= (theory_data[i] * theory_data[j])

CS = inv(th_covariance_matrix + exp_covariance_matrix)

# EXTRACTS NONZERO EIGENVALUES
eigenvectors = np.transpose(eigenvectors) # so that eig[i] is the 'i'th eigenvector

nz_eigen = [i for i in range(len(eigenvalues)) if eigenvalues[i] > 1e-5]
eigenvalues_nz = np.array([eigenvalues[i] for i in nz_eigen])[::-1]
eigenvectors_nz = np.array([eigenvectors[i] for i in nz_eigen])[::-1]

for a in range(len(eigenvectors_nz)):
    eigenvectors_nz[a] *= math.sqrt(eigenvalues_nz[a]) # 'undo' normalisation

l = len(eigenvalues_nz)
nuisance_params = np.zeros(shape=l, dtype=float64)

for a in range(0, l):
    print("Computing NPE {0} of {1}...".format(a+1, l), end='\r')
    beta = eigenvectors_nz[a]
    nuisance_params[a] = -1 * np.einsum('i,ij,j',beta,CS,TD)
    
print("Computed all NPEs                                          ")

<<<<<<< HEAD
=======
#print(nuisance_params)
>>>>>>> aa44b3700a208b693f52a790ec707232c6ea9a69

"""
*********************************************************************************************************
UNCERTAINTIES
_________________________________________________________________________________________________________

"""

# FUNCTIONS TO RETURN THE THREE TERMS USED FOR THE UNCERTAINTIES

def t1(a, b):
    e = 1 if a == b else 0
    return e

def t2(a, b):
    beta_a = eigenvectors_nz[a]
    beta_b = eigenvectors_nz[b]
    
    e = np.einsum('i,ij,j', beta_a, CS, beta_b, optimize='optimal')

    return e

def t3(a, b):
    beta_a = eigenvectors_nz[a]
    beta_b = eigenvectors_nz[b]

    e = np.einsum('i,ij,jk,kl,l', beta_a, CS, x_matrix, CS, beta_b, optimize='optimal')

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

uncertainties_nuc = np.array([math.sqrt(abs(Z[i,i])) for i in range(l)])
uncertainties_pdf = np.array([math.sqrt(abs(Z_pdf[i,i])) for i in range(l)])
uncertainties_tot = np.array([math.sqrt(abs(Z_bar[i,i])) for i in range(l)])

# NORMALISATION
norm = 1#math.sqrt(2 * math.pi * np.var(nuisance_params))
nuisance_params /= norm
uncertainties_nuc /= norm
uncertainties_pdf /= norm
uncertainties_tot /= norm

# DUMPS MATRICES TO FILE
nuisance_params.dump("matrices/NPE_" + root + ".dat")
Z.dump("matrices/Z_" + root + ".dat")
Z_pdf.dump("matrices/ZP_" + root + ".dat")
Z_bar.dump("matrices/ZN_" + root + ".dat")
uncertainties_nuc.dump("matrices/ZNE_" + root + ".dat")
uncertainties_pdf.dump("matrices/ZPE_" + root + ".dat")
uncertainties_tot.dump("matrices/ZTE_" + root + ".dat")