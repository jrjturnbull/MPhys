"""
*********************************************************************************************************
generate_combined.py

Combines data from different experiments into a single DATA + SYSTYPE file pair (compatible with covariance.py)
    -   Reads in the supplied list of DATA and SYSTYPE files
    -   Determines the experiment with the fewest nuclear replicas
    -   Selects this number of replicas from each experiment
    -   Writes DATA and SYSTYPE files to the datafiles folder 
_________________________________________________________________________________________________________

"""

import sys
import os.path
import random
import numpy as np

print()
print("Generating combined data file")

# CHECKS THAT ARGUMENTS HAVE BEEN SUPPLIED
if (len(sys.argv) < 2):
    print("ERROR: no arguments supplied...")
    quit()
root_list = sys.argv[1:-1]
output_root = sys.argv[-1]
output_data_path = "datafiles/DATA/DATA_" + output_root + ".dat"
output_syst_path = "datafiles/SYSTYPE/SYSTYPE_" + output_root + "_DEFAULT.dat"
output_theo_path = "datafiles/THEORY/THEORY_" + output_root + ".dat"
output_cent_path = "datafiles/THEORY/CENT_" + output_root + ".dat"


# DETERMINES THE NUMBER OF REPLICAS AND DATAPOINTS FOR EACH EXPERIMENT
n_nuis_list = []
n_dat_list = []
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

    row_start = 1 # ignore first line of DATA file
    row_end = 0
    with open(path_data) as file:
        linecount = len(file.readlines())
        row_end = linecount
    n_dat_list.append(row_end - row_start)

n_nuis_min = min(n_nuis_list)
print("Selecting {0} uncertainties from each data file".format(n_nuis_min))

print(n_dat_list)

# GENERATE SYSTYPE FILE
with open(output_syst_path, 'w') as syst:
    syst.write(str(n_nuis_min) + '\n')
    for i in range(0, n_nuis_min):
        syst.write("{0}    ADD    NUCLEAR{1}\n".format(i+1, i))

# GENERATE DATA FILE
with open(output_data_path, 'w') as data:
    data.write("CombinedDatafile\t{0}\t{1}\n".format(n_nuis_min, sum(n_dat_list)))

    line_no = 1

    for root in root_list:
        print("Copying data from: " + root)
        path_data = "datafiles/DATA/DATA_" + root + ".dat"
        path_syst = "datafiles/SYSTYPE/SYSTYPE_" + root + "_DEFAULT.dat"
        
        #NOTE DUPLICATED CODE BY ACCIDENT - FIX NOT NEEDED THOUGH
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
        n_nuis = nuclear_end - nuclear_start

        with open(path_data) as file:
            lines = file.readlines()
            for i in range(1, len(lines)):
                values_to_write = []
                values_to_write.append(line_no)
                values_to_write += lines[i].split('\t')[1:7] # copies kinematic data etc.
                nn = lines[i].split('\t')[7:-1] # the set of uncertainties
                nn = nn[nuclear_start * 2:] # filter to only the nuclear uncertainties
                nn = [float(n) for n in nn] # str to float conversion
                
                unc_pairs = [] # contains uncertainty pairs as formatted strings
                for j in range(0, int(len(nn)/2)):
                    unc_pairs.append('{:0.12e}'.format(nn[2*j]) + '\t' + '{:0.12e}'.format(nn[(2*j)+1]))
                unc_rand = [unc_pairs[i] for i in sorted(random.sample(range(len(unc_pairs)),n_nuis_min))]
                values_to_write += unc_rand

                for v in values_to_write:
                    data.write(str(v) + '\t')
                data.write('\n')
                line_no += 1

# GENERATE THEORY FILE
with open(output_theo_path, 'w') as theo:
    for root in root_list:
        path_theo = "datafiles/THEORY/THEORY_" + root + ".dat"
        with open(path_theo) as path:
            for l in path.readlines():
                theo.write(l)

# GENERATE CENT FILE
with open(output_cent_path, 'w') as cent:
    for root in root_list:
        path_cent = "datafiles/THEORY/CENT_" + root + ".dat"
        with open(path_cent) as path:
            for l in path.readlines():
                cent.write(l)

print()
