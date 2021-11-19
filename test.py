import numpy as np

root = "CombinedData_dw"
vec = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
print(vec)