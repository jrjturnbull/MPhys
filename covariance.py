import numpy as np
import sys
from numpy.linalg import eigh
import os

print()

#######################################################################

print("Extracting covariance matrices from /covmat")

root = sys.argv[1]

covmat_C = open("covmat/" + root + "/output/tables/groups_covmat.csv").readlines()[4:]
C = np.zeros(shape=(len(covmat_C), len(covmat_C)))
for i in range(len(covmat_C)):
    C[i] = [float(c) for c in covmat_C[i].split('\t')[3:]]

covmat_CS = open("covmat/" + root + "_dw/output/tables/groups_covmat.csv").readlines()[4:]
CS = np.zeros(shape=(len(covmat_CS), len(covmat_CS)))
for i in range(len(covmat_CS)):
    CS[i] = [float(c) for c in covmat_CS[i].split('\t')[3:]]

#if ('nuclear30' in root):
#    sift = [845,846,847,848,862,863,864,865,866,880,881,882,883,884,
#        898,899,900,901,902,916,917,918,919,920,934,935,936,937,938,
#        946,947,948,949,950]
#    CS = np.delete(CS,sift,0)
#    CS = np.delete(CS,sift,1)

S = np.array(CS - C)

covmat_CSINV = open("covmat/" + root + "_dw/output/tables/groups_invcovmat.csv").readlines()[4:]
CSINV = np.zeros(shape=(len(covmat_CSINV), len(covmat_CSINV)))
for i in range(len(covmat_CSINV)):
    CSINV[i] = [float(c) for c in covmat_CSINV[i].split('\t')[3:]]

#######################################################################

print("Extracting data-theory comparison, and computing X matrix")

covmat_DT = open("covmat/" + root + "/output/tables/group_result_table.csv").readlines()[1:]
CUTS = np.zeros(shape=len(covmat_DT))
D = np.zeros(shape=len(covmat_DT))
T = np.zeros(shape=len(covmat_DT))

theory_values = np.zeros(shape=(len(covmat_DT), len(covmat_DT[0].split('\t')[5:])))
for i in range(len(D)):
    CUTS[i] = covmat_DT[i].split('\t')[2]
    D[i] = covmat_DT[i].split('\t')[3]
    T[i] = covmat_DT[i].split('\t')[4]
    theory_values[i] = covmat_DT[i].split('\t')[5:]

X = np.zeros(shape=(len(D), len(D)))
for n in range(len(theory_values[0])):
    vec = np.array([theory_values[i,n] - T[i] for i in range(len(T))])
    X += np.einsum('i,j->ij', vec, vec) / len(theory_values[0])

#######################################################################

print("Extracting k-factor data")

if (root == "nuclear"):
    cfac_paths = ["cfactor/nuclear/CF_NUCI_NTVNUDMNFe.dat","cfactor/nuclear/CF_NUCI_NTVNBDMNFe.dat",
                "cfactor/nuclear/CF_NUCI_CHORUSNUPb.dat", "cfactor/nuclear/CF_NUCI_CHORUSNBPb.dat",
                "cfactor/nuclear/CF_NUCI_DYE605.dat"] # manual dataset ordering
elif (root == "deuterium"):
    cfac_paths = ["cfactor/deuterium/CF_DEUI_BCDMSD.dat","cfactor/deuterium/CF_DEUI_NMCPD_D.dat", 
                "cfactor/deuterium/CF_DEUI_SLACD.dat","cfactor/deuterium/CF_DEUI_DYE886R_D.dat",
                "cfactor/deuterium/CF_DEUI_DYE906R_D.dat"]
elif (root == "30"):
    cfac_paths = ["cfactor/30/CF_DEU3_BCDMSD.dat","cfactor/30/CF_DEU3_NMCPD_D.dat",
                "cfactor/30/CF_DEU3_SLACD.dat",
                "cfactor/30/CF_NUC3_NTVNUDMNFe.dat","cfactor/30/CF_NUC3_NTVNBDMNFe.dat",
                "cfactor/30/CF_NUC3_CHORUSNUPb.dat", "cfactor/30/CF_NUC3_CHORUSNBPb.dat",
                "cfactor/30/CF_DEU3_DYE886R_D.dat","cfactor/30/CF_DEU3_DYE906R_D.dat",
                "cfactor/30/CF_NUC3_DYE605.dat"]
else:
    print("Error: root not recognised")

kfac = []
cuts_left = CUTS
for path in cfac_paths:
    cfac = open(path).readlines()[9:]

    uncut_points = []
    for c in range(len(cuts_left)):
        uncut_points.append(cuts_left[c])
        if (c+1 == len(cuts_left)):
            break
        elif (cuts_left[c+1] < cuts_left[c]):
            cuts_left = cuts_left[c+1:]
            break

    kfac.extend(np.take(np.array([line.split('  ')[0].split('\t')[0] for line in cfac]), uncut_points))

kfac = np.array([float(k) for k in kfac])

#######################################################################

print("Determining eigenvalues for S")

w, v = eigh(S)
eval = np.array(w)
evec = np.array(v)

#######################################################################

C.dump("matrices/C_" + root + ".dat")
S.dump("matrices/S_" + root + ".dat")
CSINV.dump("matrices/CSINV_" + root + ".dat")

D.dump("matrices/D_" + root + ".dat")
T.dump("matrices/T_" + root + ".dat")
X.dump("matrices/X_" + root + ".dat")

eval.dump("matrices/EVL_" + root + ".dat")
evec.dump("matrices/EVC_" + root + ".dat")

kfac.dump("matrices/KFAC_" + root + ".dat")