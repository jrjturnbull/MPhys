"""
*********************************************************************************************************
extract_exp.py

Extracts the experimental covariance/correlation matrices from the various computed validphys tables:
    -   Reads in all groups_covmat csv files in ExpCov
    -   Determines experimental covariance and correlation matrices
    -   Outputs in pickled format to /matrices

NOTE: CURRENTLY ONLY HARD-CODED FOR NON-ITERATED DATAFILES
_________________________________________________________________________________________________________

"""

import os
import numpy as np
import math

print()
print("Extracting experimental covariance matrices from ExpCov")

theory_values = np.load("matrices/TH_CombinedData_dw.dat", allow_pickle=True)

rootdir = "ExpCov"
table_paths = []

# ITERATE THROUGH EACH VALIDPHYS FOLDER AND LOCATE GROUPS_COVMAT PATHS
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if "groups_covmat.csv" in path:
            table_paths.append(path)
table_paths.sort() # correct ordering

# EXTRACTS THE COVARIANCE MATRICES FOR EACH PATH
cov_matrices = []
for path in table_paths:
    print("Extracting data from: " + path)
    with open(path) as covmat:
        data_rows = covmat.readlines()[4:] # ignore header rows

        cov = np.zeros(shape=(len(data_rows),len(data_rows[0].split('\t')[3:])))
        
        for i in range(0, len(data_rows)):
            row = data_rows[i].split('\t')[3:]
            row = [float(i) for i in row]
            cov[i] = row
    
    cov_matrices.append(cov)

# MERGES THE COVARIANCE MATRICES INTO BLOCK DIAGONAL FORM
print("Computing experimental covariance matrix")
dim = sum([c.shape[0] for c in cov_matrices])
experimental_covariance = np.zeros(shape=(dim, dim))
count = 0
for c in cov_matrices:
    experimental_covariance[count:c.shape[0]+count, count:c.shape[0]+count] = c
    count += c.shape[0]

# REMOVES ROWS/COLUMNS THAT CORRESPOND TO KINEMATIC CUTS
to_cut = [] # indices of rows/columns to cut
data_path = "datafiles/DATA/DATA_CombinedData_dw.dat"
with open(data_path) as data:
    data_lines = data.readlines()[1:]
    for n in range(0, dim):
        elements = data_lines[n].split('\t')[7:-1]
        elements = [float(e) for e in elements]
        if(any(v == 0 for v in elements)):
            to_cut.append(n)
experimental_covariance = np.delete(experimental_covariance, to_cut, 0)
experimental_covariance = np.delete(experimental_covariance, to_cut, 1)

# NORMALISES THE COVARIANCE MATRIX TO THE THEORETICAL PREDICTIONS
experimental_covariance_norm = np.zeros_like(experimental_covariance)
dim_cut = experimental_covariance.shape[0]
for i in range(dim_cut):
    for j in range(dim_cut):
        if experimental_covariance[i,j] == 0:
            experimental_covariance_norm[i,j] = 0
        else:
            experimental_covariance_norm[i,j] = experimental_covariance[i,j] / (theory_values[i] * theory_values[j])

# DETERMINES THE ASSOCIATED CORRELATION MATRIX
print("Computing experimental correlation matrix")
experimental_correlation = np.zeros_like(experimental_covariance)
for i in range(0, dim_cut):
    for j in range(0, dim_cut):
        norm = math.sqrt(experimental_covariance[i,i] * experimental_covariance[j,j])
        if not (norm == 0):
            experimental_correlation[i, j] = experimental_covariance[i,j] / norm

# DUMPS ALL OUTPUT TO /MATRICES
experimental_covariance.dump("matrices/ECV_" + "CombinedData_dw" + ".dat")
experimental_covariance_norm.dump("matrices/ECVN_" + "CombinedData_dw" + ".dat")
experimental_correlation.dump("matrices/ECR_" + "CombinedData_dw" + ".dat")