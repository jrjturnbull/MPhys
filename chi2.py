import numpy as np
import sys

#######################################################################

print("Extracting chi2 values from /chi2")

root = sys.argv[1]

table_paths = ["chi2/1_nonuclear/output/tables/fits_datasets_chi2_table.csv" ,
                "chi2/2_noshift/output/tables/fits_datasets_chi2_table.csv" ,
                "chi2/3_shift/output/tables/fits_datasets_chi2_table.csv" ]

datasets = []
if (root == "nuclear"):
    datasets = ["CHORUSNBPb", "CHORUSNUPb", "DYE605", "NTVNBDMNFe", "NTVNUDMNFe"]
elif (root == "deuterium"):
    datasets = ['BCDMSD', 'DYE886R', 'NMCPD', 'SLACD']
else:
    print("Error: root not recognised")

chi2 = np.zeros(shape=(len(table_paths), len(datasets)))

for i in range(len(table_paths)):
    table = open(table_paths[i]).readlines()
    
    for j in range(len(datasets)):
        for line in table:
            if datasets[j] in line:
                chi2[i,j] = float(line.split('\t')[3])

chi2.dump("matrices/CHI2_" + root + ".dat")

#######################################################################

print("Extracting t0 chi2 values from /chi2")

root = sys.argv[1]

table_paths = ["chi2/1_nonuclear_t0/output/tables/fits_datasets_chi2_table.csv" ,
                "chi2/2_noshift_t0/output/tables/fits_datasets_chi2_table.csv" ,
                "chi2/3_shift_t0/output/tables/fits_datasets_chi2_table.csv" ]

datasets = []
if (root == "nuclear"):
    datasets = ["CHORUSNBPb", "CHORUSNUPb", "DYE605", "NTVNBDMNFe", "NTVNUDMNFe"]
elif (root == "deuterium"):
    datasets = ['BCDMSD', 'DYE886R', 'NMCPD', 'SLACD']
else:
    print("Error: root not recognised")

chi2 = np.zeros(shape=(len(table_paths), len(datasets)))

for i in range(len(table_paths)):
    table = open(table_paths[i]).readlines()
    
    for j in range(len(datasets)):
        for line in table:
            if datasets[j] in line:
                chi2[i,j] = float(line.split('\t')[3])

chi2.dump("matrices/CHI2t0_" + root + ".dat")