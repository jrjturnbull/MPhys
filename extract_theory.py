"""
*********************************************************************************************************
extract_theory.py

Extracts theory_central values from the various computed validphys tables:
    -   Reads in all group_result_table csv files in dt_comparison
    -   Extracts the theory_central column
    -   Writes to datafiles/THEORY
_________________________________________________________________________________________________________

"""

print()
print("Extracting theory data from dt_comparison")

import os
import numpy as np
rootdir = "dt_comparison_internal"
table_paths = []

# ITERATE THROUGH EACH VALIDPHYS FOLDER AND LOCATE GROUP_RESULT_TABLE PATHS
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if "group_result_table.csv" in path:
            table_paths.append(path)

# ITERATE THROUGH TABLES AND COPY THEORY_CENTRAL VALUES
for path in table_paths:
    root = ""
    theory_values = []
    cent_values = []

    with open(path) as table:
        data_rows = table.readlines()[1:] # ignore header row
        root = data_rows[0].split('\t')[1] # find root name from table (any row would work)
        for row in data_rows:
            theory_values.append(row.split('\t')[4]) # append theory_central value
            cent_values.append(row.split('\t')[3]) # append data_central value

    theory_output = "datafiles/THEORY/THEORY_" + root + ".dat"
    with open(theory_output, 'w') as output:
        for t in theory_values:
            output.write(t + '\n')
    cent_output = "datafiles/THEORY/CENT_" + root + ".dat"
    with open(cent_output, 'w') as output:
        for d in cent_values:
            output.write(d + '\n')


# EXTRACT CFACTOR DATA
print("Extracting K factor data")
root_list = ["CHORUSNBPb_dw", "CHORUSNUPb_dw", "DYE605_dw", "NTVNBDMNFe_dw", "NTVNUDMNFe_dw"]
for root in root_list:
    cf_file = open("datafiles/CF/CF_NUC_" + root + ".dat").readlines()
    cuts = [int(c) for c in open("datafiles/CUTS/CUTS_" + root + ".dat").readlines()]

    x = 0
    while(True): # ignore header lines
        x = x+1 if "***" in cf_file[0] else x
        cf_file = cf_file[1:]
        if (x==2):
            break

    cf_file = np.array(cf_file)
    cf_file = np.delete(cf_file, cuts)

    k_file = open("datafiles/CF/K_" + root + ".dat", 'w')
    k_file.writelines([line.split('  ')[0]+'\n' for line in cf_file])
    