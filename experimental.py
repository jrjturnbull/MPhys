import numpy as np
import math
from numpy import exp, float64
import sys
import os

# CHECKS THAT ARGUMENTS HAVE BEEN SUPPLIED
if (len(sys.argv) < 2):
    print("ERROR: no arguments supplied...")
    quit()
root_list = sys.argv[1:-1]
output_root = sys.argv[-1]

# DETERMINES THE NUMBER OF REPLICAS AND DATAPOINTS FOR EACH EXPERIMENT
n_nuis_list = []
n_dat_list = []
ns_list = []
for root in root_list:
    path_data = "datafiles/DATA/DATA_" + root + ".dat"
    path_syst = "datafiles/SYSTYPE/SYSTYPE_" + root + "_DEFAULT.dat"
    path_theo = "datafiles/THEORY/THEORY_" + root + ".dat"

    if not os.path.exists(path_data):
        print("ERROR: " + path_data + " does not exist!")
        quit()
    if not os.path.exists(path_syst):
        print("ERROR: " + path_syst + " does not exist!")
        quit()
    if not os.path.exists(path_theo):
        print("ERROR: " + path_theo + " does not exist!")
        quit()

    nuclear_start = 0
    nuclear_end = 0
    with open(path_syst) as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            entries = lines[i].split("    ")
            if ("NUCLEAR" in entries[2]):
                nuclear_start = int(entries[0]) - 1
                break
        nuclear_end = len(lines) - 1 # assumes that final uncertainty is always nuclear
    n_nuis_list.append(nuclear_end - nuclear_start)
    ns_list.append(nuclear_start)

    row_start = 1 # ignore first line of DATA file
    row_end = 0
    with open(path_data) as file:
        linecount = len(file.readlines())
        row_end = linecount
    n_dat_list.append(row_end - row_start)

exp_covariance_matrix_list = []
exp_correlation_matrix_list = []
dim = 0
for r in range(len(root_list)):
    experimental_unc = np.zeros(shape = n_dat_list[r], dtype=float64)
    zero_lines = 0 # records how many zero lines have been found

    path_data = "datafiles/DATA/DATA_" + root_list[r] + ".dat"
    with open(path_data) as data:
        data_lines = data.readlines()
        for i in range(1, n_dat_list[r]+1):
            nuclear_uncertainties = data_lines[i].split("\t")[7:-1:2] # remove exp. values and mult. uncertanties
            ns = ns_list[r]
            nuclear_uncertainties = nuclear_uncertainties[ns:] # remove non-nuclear uncertainties
            nuclear_uncertainties = [float64(i) for i in nuclear_uncertainties] # convert from str to f64

            # Remove rows containing all zeros
            if(all(v == 0 for v in nuclear_uncertainties)):
                zero_lines += 1 # records that a zero line has been found
                experimental_unc = experimental_unc[:-1]
                continue
            
            experimental_unc[(i-1) - zero_lines] = data_lines[i].split("\t")[6]
    
    n_dat_nz = len(experimental_unc) # number of non-zero data points

    exp_matrix = np.zeros(shape = (n_dat_nz, n_dat_nz))
    for x in range(n_dat_nz):
        for y in range(n_dat_nz):
            exp_matrix[x,y] = experimental_unc[x] * experimental_unc[y]
    exp_covariance_matrix_list.append(exp_matrix)
    dim += n_dat_nz

    # DETERMINE THE CORRELATION MATRIX
    correlation_matrix = np.zeros_like(exp_matrix)
    for x in range(0, n_dat_nz):
        for y in range(0, n_dat_nz):
            norm = math.sqrt(exp_matrix[x,x] * exp_matrix[y,y])
            correlation_matrix[x, y] = exp_matrix[x,y] / norm
    exp_correlation_matrix_list.append(correlation_matrix)

exp_covariance_matrix = np.zeros(shape=(dim, dim), dtype=float64)
exp_correlation_matrix = np.zeros(shape=(dim, dim), dtype=float64)
dat_count = 0
for e in range(len(exp_covariance_matrix_list)):
    nd = exp_covariance_matrix_list[e].shape[0]
    exp_covariance_matrix[dat_count:(dat_count+nd), dat_count:(dat_count+nd)] = exp_covariance_matrix_list[e]
    exp_correlation_matrix[dat_count:(dat_count+nd), dat_count:(dat_count+nd)] = exp_correlation_matrix_list[e]
    dat_count += nd

exp_covariance_matrix.dump("matrices/ECV_" + output_root + ".dat")
exp_correlation_matrix.dump("matrices/ECR_" + output_root + ".dat")