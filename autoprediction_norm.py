from math import sqrt
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

root = "CombinedData_dw"

th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
x_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

# NORMALISATION TO THEORY (JUST IN CASE THIS MAGICALLY SOLVES EVERYTHING...)
th_covariance_matrix = np.reshape(np.array([th_covariance_matrix[i,j] / (theory_data[i] * theory_data[j]) \
    for i in range(len(th_covariance_matrix)) for j in range(len(th_covariance_matrix))]), \
    (len(th_covariance_matrix), len(th_covariance_matrix)))

exp_covariance_matrix = np.reshape(np.array([exp_covariance_matrix[i,j] / (theory_data[i] * theory_data[j]) \
    for i in range(len(exp_covariance_matrix)) for j in range(len(exp_covariance_matrix))]), \
    (len(exp_covariance_matrix), len(exp_covariance_matrix)))

x_matrix = np.reshape(np.array([x_matrix[i,j] / (theory_data[i] * theory_data[j]) \
    for i in range(len(x_matrix)) for j in range(len(x_matrix))]), \
    (len(x_matrix), len(x_matrix)))

# DETERMINE AUTOPREDICTION SHIFTS
CS = inv(exp_covariance_matrix + th_covariance_matrix)
SCS = np.einsum('ij,jk->ik', th_covariance_matrix, CS)
TD = (theory_data - exp_data) / theory_data
delta_T = -np.einsum('ij,j->i', SCS, TD)

delta_T.dump("matrices/DT_CombinedData_dw.dat")
TD.dump("matrices/TD_CombinedData_dw.dat")



# DETERMINE AUTOPREDICTION COVARIANCE MATRIX
term_1 = np.einsum('ij,jk,kl,lm,mn->in', exp_covariance_matrix, CS, x_matrix, CS, exp_covariance_matrix, optimize='optimal')
term_2 = th_covariance_matrix - np.einsum('ij,jk,kl->il', th_covariance_matrix, CS, th_covariance_matrix, optimize='optimal')
autoprediction = term_1 + term_2

autoprediction.dump("matrices/AP_CombinedData_dw.dat")


# DETERMINE CONTRIBUTIONS TO THE DIAGONAL ELEMENTS OF THE CORRELATED THEORY & PDF UNCERTAINTIES
th_contribution_1 = th_covariance_matrix - np.einsum('ij,jk,kl->il', th_covariance_matrix,CS, th_covariance_matrix, optimize='optimal')
th_contribution_2 = th_contribution_1 + np.einsum('ij,jk,kl,lm,mn->in', th_covariance_matrix, CS, x_matrix, CS, th_covariance_matrix, optimize='optimal')

x_contribution_1 = np.einsum('ij,jk,kl,lm,mn->in', exp_covariance_matrix, CS, x_matrix, CS, exp_covariance_matrix, optimize='optimal')
x_contribution_2 = x_matrix - np.einsum('ij,jk,kl->il', th_covariance_matrix,CS,x_matrix, optimize='optimal')       \
     - np.einsum('ij,jk,kl->il', x_matrix,CS, th_covariance_matrix, optimize='optimal')

th_contribution_1.dump("matrices/TH1_CombinedData_dw.dat")
th_contribution_2.dump("matrices/TH2_CombinedData_dw.dat")
x_contribution_1.dump("matrices/X1_CombinedData_dw.dat")
x_contribution_2.dump("matrices/X2_CombinedData_dw.dat")