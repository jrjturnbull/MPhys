import numpy as np

data = open("datafiles/DATA/DATA_CombinedData_dw.dat").readlines()[1:]
cuts = [int(i) for i in open("datafiles/CUTS/CUTS_CombinedData_dw.dat").readlines()]

nrep = np.sqrt(np.concatenate((np.ones(416)*100, np.ones(416)*1000, np.ones(85)*1000, np.ones(37)*100, np.ones(39)*1000)))

nuclear = np.zeros(shape=len(data)-len(cuts))

add = 0
for i in range(len(data)):
    if (i in cuts):
        add += 1
        continue
    datapoints = np.array([float(d) for d in data[i].split('\t')[7:-1:2]])
    nuclear[i-add] = np.average(datapoints)

nuclear *= nrep

nuclear.dump("matrices/NUC_CombinedData_dw.dat")