import numpy as np
import sys

root = sys.argv[1]

C = np.load("matrices/C_" + root + ".dat", allow_pickle=True)
S = np.load("matrices/S_" + root + ".dat", allow_pickle=True)
invCS = np.load("matrices/CSINV_" + root + ".dat", allow_pickle=True)

D = np.load("matrices/D_" + root + ".dat", allow_pickle=True)
T = np.load("matrices/T_" + root + ".dat", allow_pickle=True)
X = np.load("matrices/X_" + root + ".dat", allow_pickle=True)
TD = T - D

#######################################################################

print("Determining autoprediction matrix + shifts")

term_1 = np.einsum('ij,jk,kl,lm,mn->in', C, invCS, X, invCS, C, optimize='optimal')
term_2 = S - np.einsum('ij,jk,kl->il', S, invCS, S, optimize='optimal')
P = term_1 + term_2

delta_T = - np.einsum('ij,jk,k->i', S, invCS, TD)

#######################################################################

print("Determining covariance matrix diagonal contributions")

S_contribution_1 = S - np.einsum('ij,jk,kl->il', S, invCS, S, optimize='optimal')
S_contribution_2 = S_contribution_1 + np.einsum('ij,jk,kl,lm,mn->in', S, invCS, X, invCS, S, optimize='optimal')

X_contribution_1 = np.einsum('ij,jk,kl,lm,mn->in', C, invCS, X, invCS, C, optimize='optimal')
X_contribution_2 = X - np.einsum('ij,jk,kl->il', S, invCS, X, optimize='optimal')       \
    - np.einsum('ij,jk,kl,lm,mn->in', S, invCS, X, invCS, S, optimize='optimal')


#######################################################################

P.dump("matrices/P_" + root + ".dat")
delta_T.dump("matrices/AUTO_" + root + ".dat")

S_contribution_1.dump("matrices/S1_" + root + ".dat")
S_contribution_2.dump("matrices/S2_" + root + ".dat")
X_contribution_1.dump("matrices/X1_" + root + ".dat")
X_contribution_2.dump("matrices/X2_" + root + ".dat")