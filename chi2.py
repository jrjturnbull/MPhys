import numpy as np
from numpy.linalg import inv

root = "CombinedData_dw"

th_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
exp_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
x_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

CS = inv(exp_covariance_matrix + th_covariance_matrix)
TD = theory_data - exp_data

# DETERMINE CHI2 VALUES
CS_shifted = np.zeros_like(CS)
with open("ExpCov/CS_shifted/output/tables/groups_covmat.csv") as file:
     lines = file.readlines()[4:]
     CS_shifted = np.array([lines[i].split("\t")[3:] for i in range(len(lines))], dtype=float)

ds = [0,416,832,917,954,len(TD)] # shorthand for dataset_split
chi2_no_th = np.zeros(len(ds)-1)
chi2_yes_th = np.zeros(len(ds)-1)
chi2_shifted = np.zeros(len(ds)-1)
chi2_auto = np.zeros(len(ds)-1)

for n in range(len(ds) - 1):
     TD_ds = TD[ds[n]:ds[n+1]]
     C_ds = exp_covariance_matrix[ds[n]:ds[n+1],ds[n]:ds[n+1]]
     S_ds = th_covariance_matrix[ds[n]:ds[n+1],ds[n]:ds[n+1]]
     CS_ds = inv((exp_covariance_matrix+th_covariance_matrix)[ds[n]:ds[n+1],ds[n]:ds[n+1]])
     CS_shifted_ds = inv((CS_shifted)[ds[n]:ds[n+1],ds[n]:ds[n+1]])

     delta_T_ds = - np.einsum('ij,jk,k->i', S_ds, CS_ds, TD_ds)
     theory_data_ds = theory_data[ds[n]:ds[n+1]]
     auto_ds = delta_T_ds # + TD_ds

     chi2_no_th[n] = np.einsum('i,ij,j', TD_ds, inv(C_ds), TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     chi2_yes_th[n] = np.einsum('i,ij,j', TD_ds, CS_ds, TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     chi2_shifted[n] = np.einsum('i,ij,j', TD_ds, CS_shifted_ds, TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     chi2_auto[n] = np.einsum('i,ij,j', auto_ds, CS_ds, auto_ds, optimize='optimal') / (ds[n+1] - ds[n])


# DETERMINE CHI2 VALUES FOR t0 METHOD
CS_t0 = np.zeros_like(CS)
with open("ExpCov/t0_CS/output/tables/groups_covmat.csv") as file:
     lines = file.readlines()[4:]
     CS_t0 = np.array([lines[i].split("\t")[3:] for i in range(len(lines))], dtype=float)

C_t0 = np.zeros_like(CS)
with open("ExpCov/t0_Nuclear/output/tables/groups_covmat.csv") as file:
     lines = file.readlines()[4:]
     C_t0 = np.array([lines[i].split("\t")[3:] for i in range(len(lines))], dtype=float)

CS_shifted_t0 = np.zeros_like(CS)
with open("ExpCov/t0_CS_shifted/output/tables/groups_covmat.csv") as file:
     lines = file.readlines()[4:]
     CS_shifted_t0 = np.array([lines[i].split("\t")[3:] for i in range(len(lines))], dtype=float)

ds = [0,416,832,917,954,len(TD)] # shorthand for dataset_split
chi2_no_th_t0 = np.zeros(len(ds)-1)
chi2_yes_th_t0 = np.zeros(len(ds)-1)
chi2_shifted_t0 = np.zeros(len(ds)-1)
#chi2_auto_t0 = np.zeros(len(ds)-1)

for n in range(len(ds) - 1):
     TD_ds = TD[ds[n]:ds[n+1]]

     C_ds_t0 = C_t0[ds[n]:ds[n+1],ds[n]:ds[n+1]]
     CS_ds_t0 = inv(CS_t0[ds[n]:ds[n+1],ds[n]:ds[n+1]])
     CS_shifted_ds_t0 = inv(CS_shifted_t0[ds[n]:ds[n+1],ds[n]:ds[n+1]])

     chi2_no_th_t0[n] = np.einsum('i,ij,j', TD_ds, inv(C_ds_t0), TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     chi2_yes_th_t0[n] = np.einsum('i,ij,j', TD_ds, CS_ds_t0, TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     chi2_shifted_t0[n] = np.einsum('i,ij,j', TD_ds, CS_shifted_ds_t0, TD_ds, optimize='optimal') / (ds[n+1] - ds[n])
     #chi2_auto_t0[n] = np.einsum('i,ij,j', auto_ds, CS_ds, auto_ds, optimize='optimal') / (ds[n+1] - ds[n])


chi2_no_th.dump("matrices/CHN_CombinedData_dw.dat")
chi2_yes_th.dump("matrices/CHY_CombinedData_dw.dat")
chi2_shifted.dump("matrices/CHS_CombinedData_dw.dat")
chi2_auto.dump("matrices/CHA_CombinedData_dw.dat")

chi2_no_th_t0.dump("matrices/CHNt0_CombinedData_dw.dat")
chi2_yes_th_t0.dump("matrices/CHYt0_CombinedData_dw.dat")
chi2_shifted_t0.dump("matrices/CHSt0_CombinedData_dw.dat")