import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
import numpy as np

# ATTEMPT TO REPLICATE THE COLORBAR USED IN THE LITERATURE (STILL NOT QUITE RIGHT...)
c = ["firebrick","red","chocolate","orange","sandybrown","peachpuff","lightyellow",
        "honeydew","palegreen","aquamarine","mediumturquoise", "royalblue","midnightblue"]
v = [0,.1,.2,.3,.4,.45,.5,.55,.6,.7,.8,.9,1]
l = list(zip(v,reversed(c)))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

C = np.load("matrices/ECV_CombinedData_dw.dat", allow_pickle=True)
S = np.load("matrices/CV_CombinedData_dw.dat", allow_pickle=True)
THEORY = np.load("matrices/TH_CombinedData_dw.dat", allow_pickle=True)
CS_manual = C + S

cs_path = "ExpCov/CS/output/tables/groups_covmat.csv"
n_dat = sum(1 for line in open(cs_path)) - 4
CS_validphys = np.zeros(shape=(n_dat, n_dat))
with open(cs_path) as cs:
    lines = cs.readlines()[4:]
    for i in range(len(lines)):
        CS_validphys[i] = lines[i].split("\t", )[3:]

plt.imshow(C, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("C from ValidPhys")
plt.savefig("CS_validphys")
plt.clf()

plt.imshow(S, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("S from ValidPhys")
plt.savefig("CS_manual")
plt.clf()