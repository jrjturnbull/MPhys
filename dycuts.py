import numpy as np

covmat_old = open("covmat/nuclear/output/tables/groups_covmat.csv").readlines()
covmat_30 = open("covmat/nuclear30_dw/output/tables/groups_covmat.csv").readlines()

id_old = np.zeros(shape=len(covmat_old)-4)
for i in range(len(id_old)):
    id_old[i] = covmat_old[i+4].split('\t')[2]

#print(id_old)

id_30 = np.zeros(shape=len(covmat_30)-4)
for i in range(len(id_30)):
    id_30[i] = covmat_30[i+4].split('\t')[2]

ids = np.array([13,14,15,16,30,31,32,33,34,48,49,50,51,52,66,67,68,69,70,84,85,86,87,
                    88,102,103,104,105,106,114,115,116,117,118])


sift = [845,846,847,848,862,863,864,865,866,880,881,882,883,884,
        898,899,900,901,902,916,917,918,919,920,934,935,936,937,938,
        946,947,948,949,950]

ids_sifted = id_30[sift]
#print(ids_sifted == [int(id) for id in ids])

print(id_30[0])