"""
*********************************************************************************************************
extract_theory.py

Extracts theory_central values from the various computed validphys tables:
    -   Reads in group_result_table.csv for supplied validphys folder
    -   Extracts the theory_central column
    -   Writes to datafiles/THEORY
_________________________________________________________________________________________________________

"""

import os
rootdir = "dt_comparison"
table_paths = []

# ITERATE THROUGH EACH VALIDPHYS FOLDER AND LOCATE GROUP_RESULT_TABLE PATHS
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        if "group_result_table.csv" in path:
            table_paths.append(path)

# ITERATE THROUGH TABLES AND COPY THEORY_CENTRAL VALUES
for path in table_paths:
    root = ""
    theory_values = []
    
    with open(path) as table:
        data_rows = table.readlines()[1:] # ignore header row
        root = data_rows[0].split('\t')[1] # find root name from table (any row would work)
        for row in data_rows:
            theory_values.append(row.split('\t')[4]) # append theory_central value
    
    output_path = "datafiles/THEORY/THEORY_" + root + ".dat"
    with open(output_path, 'w') as output:
        for t in theory_values:
            output.write(t + '\n')