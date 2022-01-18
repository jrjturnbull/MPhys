"""
*********************************************************************************************************
pdf_covariance.py

Computes the pdf covariance/correlation X matrices
_________________________________________________________________________________________________________

"""

from matplotlib.colors import LogNorm
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import math

rootdir = "dt_comparison_internal"
table_paths = []

# ITERATE THROUGH EACH VALIDPHYS FOLDER AND LOCATE GROUP_RESULT_TABLE PATHS
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if "group_result_table.csv" in path:
            table_paths.append(path)
table_paths.sort()

# ONLY DEAL WITH NON-ITERATED FILES FOR NOW...
for t in table_paths:
    if "ite" in t:
        table_paths.remove(t)

# READ IN THEORY VALUES
theory_values = []
central_values = []
for i in range(0, len(table_paths)):
    with open(table_paths[i]) as theory_table:
        lines = theory_table.readlines()[1:]
        for l in lines:
            values = l.split('\t')[4:]
            values = [float(v) for v in values]
            theory_values.append(values[5:-1])
            central_values.append(values[4])

data_path = "datafiles/DATA/DATA_CombinedData_dw.dat"
with open(data_path) as data:
    data_lines = data.readlines()[1:]
    for n in range(0, len(data_lines)):
        elements = data_lines[n].split('\t')[7:-1]
        elements = [float(e) for e in elements]
central_values = np.array(central_values)
theory_values = np.array(theory_values)

# DETERMINE THE COVARIANCE MATRIX
n_dat = len(central_values)
n_nuis = len(theory_values[0])

x_matrix = np.zeros(shape=(n_dat,n_dat))
for n in range(n_nuis):
    x_vector = np.array([theory_values[i,n] - central_values[i] for i in range(n_dat)])
    x_matrix += np.einsum('i,j->ij', x_vector, x_vector) / n_nuis

# DETERMINE THE CORRELATION MATRIX
dim = x_matrix.shape[0]
correlation = np.zeros_like(x_matrix)
for i in range(0, dim):
    for j in range(0, dim):
        norm = math.sqrt(x_matrix[i,i] * x_matrix[j,j])
        correlation[i, j] = x_matrix[i,j] / norm

# SAVE OUTPUT
x_matrix.dump("matrices/XCV_" + "CombinedData_dw" + ".dat")
correlation.dump("matrices/XCR_" + "CombinedData_dw" + ".dat")

# fig, ax = plt.subplots()
# im = ax.imshow(correlation, cmap='jet', vmin=-1, vmax=1)
# plt.colorbar(im)
# plt.show()