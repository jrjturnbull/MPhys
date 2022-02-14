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

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

print()
print("Extracting experimental covariance matrices from ExpCov")

table_path = "ExpCov/Nuclear/output/tables/groups_covmat.csv"

# EXTRACTS THE COVARIANCE MATRICES FOR EACH PATH
covmat =  open(table_path)
data_rows = covmat.readlines()[4:] # ignore header rows
experimental_covariance = np.zeros(shape=(len(data_rows),len(data_rows)))

for i in range(0, len(data_rows)):
    row = data_rows[i].split('\t')[3:]
    row = [float(i) for i in row]
    experimental_covariance[i] = row

# DETERMINES THE ASSOCIATED CORRELATION MATRIX
print("Computing experimental correlation matrix")
experimental_correlation = np.zeros_like(experimental_covariance)
for i in range(0, len(data_rows)):
    for j in range(0, len(data_rows)):
        norm = math.sqrt(experimental_covariance[i,i] * experimental_covariance[j,j])
        experimental_correlation[i, j] = experimental_covariance[i,j] / norm

# DUMPS ALL OUTPUT TO /MATRICES
experimental_covariance.dump("matrices/ECV_" + "CombinedData_dw" + ".dat")
experimental_correlation.dump("matrices/ECR_" + "CombinedData_dw" + ".dat")
