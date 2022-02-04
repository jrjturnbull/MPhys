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