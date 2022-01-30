import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
import numpy as np
import math

# ATTEMPT TO REPLICATE THE COLORBAR USED IN THE LITERATURE (STILL NOT QUITE RIGHT...)
c = ["firebrick","red","chocolate","orange","sandybrown","peachpuff","lightyellow",
        "honeydew","palegreen","aquamarine","mediumturquoise", "royalblue","midnightblue"]
v = [0,.1,.2,.3,.4,.45,.5,.55,.6,.7,.8,.9,1]
l = list(zip(v,reversed(c)))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

C = np.load("matrices/CV_CombinedData_dw.dat", allow_pickle=True)
S = np.load("matrices/ECV_CombinedData_dw.dat", allow_pickle=True)

CS_manual = np.linalg.inv(C+S)

lines = open("ExpCov/CS/output/tables/groups_invcovmat.csv").readlines()[4:]
CS_validphys = np.array([lines[i].split("\t")[3:] for i in range(len(CS_manual))], dtype=float)

plt.imshow(CS_manual, cmap=cmap, norm=SymLogNorm(1e-4))
plt.colorbar()
plt.savefig("CS_manual")

plt.clf()
plt.imshow(CS_validphys, cmap=cmap, norm=SymLogNorm(1e-4))
plt.colorbar()
plt.savefig("CS_validphys")