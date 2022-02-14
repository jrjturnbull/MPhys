import numpy as np
import sys
from numpy.linalg import eigh

print()

#######################################################################

print("Extracting covariance matrices from /covmat")

root = sys.argv[1]

covmat_C = open("covmat/" + root + "/output/tables/groups_covmat.csv").readlines()[4:]
C = np.zeros(shape=(len(covmat_C), len(covmat_C)))
for i in range(len(covmat_C)):
    C[i] = [float(c) for c in covmat_C[i].split('\t')[3:]]

covmat_CS = open("covmat/" + root + "_dw/output/tables/groups_covmat.csv").readlines()[4:]
CS = np.zeros(shape=(len(covmat_CS), len(covmat_CS)))
for i in range(len(covmat_CS)):
    CS[i] = [float(c) for c in covmat_CS[i].split('\t')[3:]]

S = np.array(CS - C)

covmat_CSINV = open("covmat/" + root + "_dw/output/tables/groups_invcovmat.csv").readlines()[4:]
CSINV = np.zeros(shape=(len(covmat_CSINV), len(covmat_CSINV)))
for i in range(len(covmat_CSINV)):
    CSINV[i] = [float(c) for c in covmat_CSINV[i].split('\t')[3:]]

#######################################################################

print("Extracting data-theory comparison, and computing X matrix")

covmat_DT = open("covmat/" + root + "/output/tables/group_result_table.csv").readlines()[1:]
D = np.zeros(shape=len(covmat_DT))
T = np.zeros(shape=len(covmat_DT))
theory_values = np.zeros(shape=(len(covmat_DT), len(covmat_DT[0].split('\t')[5:])))
for i in range(len(D)):
    D[i] = covmat_DT[i].split('\t')[3]
    T[i] = covmat_DT[i].split('\t')[4]
    theory_values[i] = covmat_DT[i].split('\t')[5:]

X = np.zeros(shape=(len(D), len(D)))
for n in range(len(theory_values[0])):
    vec = np.array([theory_values[i,n] - T[i] for i in range(len(T))])
    X += np.einsum('i,j->ij', vec, vec) / len(theory_values[0])

#######################################################################

print("Determining eigenvalues for S")

w, v = eigh(S)
eval = np.array(w)
evec = np.array(v)

#######################################################################

C.dump("matrices/C_" + root + ".dat")
S.dump("matrices/S_" + root + ".dat")
CSINV.dump("matrices/CSINV_" + root + ".dat")

D.dump("matrices/D_" + root + ".dat")
T.dump("matrices/T_" + root + ".dat")
X.dump("matrices/X_" + root + ".dat")

eval.dump("matrices/EVL_" + root + ".dat")
evec.dump("matrices/EVC_" + root + ".dat")