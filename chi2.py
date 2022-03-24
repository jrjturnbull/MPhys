import numpy as np
import sys

#######################################################################

print("Extracting chi2 values from /chi2")

root = sys.argv[1]

table_paths = ["chi2/"+root+"/1_nonuclear/output/tables/groups_chi2_table.csv" ,
                "chi2/"+root+"/2_noshift/output/tables/groups_chi2_table.csv" ,
                "chi2/"+root+"/3_shift/output/tables/groups_chi2_table.csv" ]

datasets = []
if (root == "nuclear"):
    datasets = ["NTVNUDMNFe", "NTVNBDMNFe", "CHORUSNUPb", "CHORUSNBPb", "DYE605"]
elif (root == "deuterium"):
    datasets = ['BCDMSD', 'NMCPD', 'SLACD', 'DYE886R','DYE906R']
elif (root == "30"):
    datasets = ['BCDMSD', 'NMCPD', 'SLACD', "NTVNUDMNFe", "NTVNBDMNFe", "CHORUSNUPb", "CHORUSNBPb", 'DYE886R','DYE906R', "DYE605"]
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

table_paths = ["chi2/"+root+"/1_nonuclear_t0/output/tables/groups_chi2_table.csv" ,
                "chi2/"+root+"/2_noshift_t0/output/tables/groups_chi2_table.csv" ,
                "chi2/"+root+"/3_shift_t0/output/tables/groups_chi2_table.csv" ]

datasets = []
if (root == "nuclear"):
    datasets = ["NTVNUDMNFe", "NTVNBDMNFe", "CHORUSNUPb", "CHORUSNBPb", "DYE605"]
elif (root == "deuterium"):
    datasets = ['BCDMSD', 'NMCPD', 'SLACD', 'DYE886R','DYE906R']
elif (root == "30"):
    datasets = ['BCDMSD', 'NMCPD', 'SLACD', "NTVNUDMNFe", "NTVNBDMNFe", "CHORUSNUPb", "CHORUSNBPb", 'DYE886R','DYE906R', "DYE605"]
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

#######################################################################

from numpy.linalg import inv

print("Manual chi2 computation......")
T = np.load("matrices/T_" + root + ".dat", allow_pickle=True)[0:248]
D = np.load("matrices/D_" + root + ".dat", allow_pickle=True)[0:248]
invC = inv(np.load("matrices/C_" + root + ".dat", allow_pickle=True)[0:248, 0:248])

chi2_manual = np.einsum("i,ij,j", (T-D), invC, (T-D)) / 248
print(chi2_manual)

T = np.load("matrices/T_" + root + ".dat", allow_pickle=True)[248:369]
D = np.load("matrices/D_" + root + ".dat", allow_pickle=True)[248:369]
invC = inv(np.load("matrices/C_" + root + ".dat", allow_pickle=True)[248:369, 248:369])

chi2_manual = np.einsum("i,ij,j", (T-D), invC, (T-D)) / 121
print(chi2_manual)

print(chi2)