print()
print("Extracting kinematic cuts from dt_comparison_internal")

import os
import numpy as np

dir_nocuts = "dt_comparison"
table_paths_nocuts = []
dir_internal = "dt_comparison_internal"
table_paths_internal = []
roots = []

for subdir, dirs, files in os.walk(dir_nocuts):
    for file in files:
        path = os.path.join(subdir, file)
        if "group_result_table.csv" in path:
            if "ite" not in path:
                table_paths_nocuts.append(path)
for t in table_paths_nocuts:
    if "ite" in t:
        table_paths_nocuts.remove(t)
line_numbers_nocuts = []
for path in table_paths_nocuts:
    print(path)
    data = open(path)
    data.readline()
    line_numbers = []
    lines = data.readlines()
    roots.append(lines[0].split('\t')[1])
    for line in lines:
        line_numbers.append(line.split('\t')[2])
    line_numbers_nocuts.append(line_numbers)

for subdir, dirs, files in os.walk(dir_internal):
    for file in files:
        path = os.path.join(subdir, file)
        if "group_result_table.csv" in path:
            if "ite" not in path:
                table_paths_internal.append(path)
line_numbers_internal = []
for path in table_paths_internal:
    print(path)
    data = open(path)
    data.readline()
    line_numbers = []
    for line in data.readlines():
        line_numbers.append(line.split('\t')[2])
    line_numbers_internal.append(line_numbers)

for i in range(len(roots)):
    c = np.sort(np.setdiff1d(line_numbers_nocuts[i], line_numbers_internal[i]).astype(int))
    output = open("datafiles/CUTS/CUTS_" + roots[i] + ".dat", 'w')
    for n in c:
        output.write(str(n) + "\n")