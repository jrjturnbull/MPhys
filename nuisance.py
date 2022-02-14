import numpy as np
import math
import sys

root = sys.argv[1]

C = np.load("matrices/C_" + root + ".dat", allow_pickle=True)
S = np.load("matrices/S_" + root + ".dat", allow_pickle=True)
invCS = np.load("matrices/CSINV_" + root + ".dat", allow_pickle=True)

D = np.load("matrices/D_" + root + ".dat", allow_pickle=True)
T = np.load("matrices/T_" + root + ".dat", allow_pickle=True)
X = np.load("matrices/X_" + root + ".dat", allow_pickle=True)
TD = T - D

eval = np.load("matrices/EVAL_" + root + ".dat", allow_pickle=True)
evec = np.load("matrices/EVEC_" + root + ".dat", allow_pickle=True)

evec = np.transpose(evec) # so that evec[i] is the 'i'th eigenvector
nz_eigen = [i for i in range(len(eval)) if eval[i] > 1e-5]
eval_nz = np.array([eval[i] for i in nz_eigen])[::-1]
evec_nz = np.array([evec[i] for i in nz_eigen])[::-1]
for a in range(len(evec_nz)):
    evec_nz[a] *= math.sqrt(eval_nz[a])
l = len(eval_nz)

NPE = np.zeros(shape=l)
for a in range(l):
    print("Computing NPE {0} of {1}...".format(a+1,l), end="\r")
    beta = evec_nz[a]
    NPE[a] = -1 * np.einsum('i,ij,j', beta, invCS, TD)
print("Computed all NPEs                               ")

#######################################################################

def t1(a,b):
    e = 1 if a == b else 0
    return e

def t2(a,b):
    beta_a = evec_nz[a]
    beta_b = evec_nz[b]
    e = np.einsum('i,ij,j', beta_a, invCS, beta_b, optimize='optimal')
    return e

def t3(a,b):
    beta_a = evec_nz[a]
    beta_b = evec_nz[b]
    e = np.einsum('i,ij,jk,kl,l', beta_a, invCS, X, invCS, beta_b, optimize='optimal')
    return e

Z = np.zeros(shape=(l,l))
Z_bar = np.zeros(shape=(l,l))
Z_pdf = np.zeros(shape=(l,l))
for a in range(l):
    print("Computing NPE uncertainty {0} of {1}...".format(a+1,l), end="\r")
    for b in range(l):
        t_1 = t1(a,b)
        t_2 = t2(a,b)
        t_3 = t3(a,b)

        Z[a,b] = t_1 - t_2
        Z_pdf[a,b] = -t_3
        Z_bar[a,b] = t_1 - t_2 - t_3

NPE_nuc = np.array([math.sqrt(Z[i,i]) for i in range(l)])
NPE_pdf = np.array([math.sqrt(Z_pdf[i,i]) for i in range(l)])
NPE_tot = np.array([math.sqrt(Z_bar[i,i]) for i in range(l)])

print("Computed all NPE uncertainties                                           ")

NPE.dump("matrices/NPE_" + root + ".dat")
NPE_nuc.dump("matrices/NPEnuc_" + root + ".dat")
NPE_pdf.dump("matrices/NPEpdf_" + root + ".dat")
NPE_tot.dump("matrices/NPEtot_" + root + ".dat")