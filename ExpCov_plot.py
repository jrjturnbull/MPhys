from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

path = "ExpCov/output/tables/groups_covmat.csv"

# FIND NUMBER OF DATAPOINTS
row_start = 4 # ignore first four lines
row_end = 0
with open(path) as file:
    linecount = len(file.readlines())
    row_end = linecount
n_dat = row_end - row_start

heatmap = np.zeros(shape=(n_dat, n_dat))

with open(path) as covmat:
    lines = covmat.readlines()[row_start : row_end]
    for i in range(n_dat):
        for j in range(n_dat):
            heatmap[i,j] = lines[i].split("\t")[2 + j]

fig, ax = plt.subplots()
im = ax.imshow(heatmap, cmap='jet', norm=LogNorm())
plt.colorbar(im)
plt.show()