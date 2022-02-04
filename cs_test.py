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

<<<<<<< HEAD

S = np.load("matrices/CV_CombinedData_dw.dat", allow_pickle=True)
evec = np.load("matrices/EVCN_CombinedData_dw.dat", allow_pickle=True)
eval = np.load("matrices/EVLN_CombinedData_dw.dat", allow_pickle=True)

S_manual = np.zeros_like(S)
for a in range(len(evec)):
        vec = evec[:,a] * math.sqrt(abs(eval[a]))
        S_manual += np.outer(vec, vec)

plt.imshow(S, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("S correct")
plt.show()
plt.clf()

plt.imshow(S_manual, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("S manual")
plt.show()
plt.clf()
=======
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
>>>>>>> aa44b3700a208b693f52a790ec707232c6ea9a69
