# MANUAL CHECK OF COVARIANCE MATRIX ELEMENTS - CAN BE IGNORED

path = "datafiles/DATA_NTVNBDMNFe_dw_ite.dat"

exp = 0
nn = []

with open(path) as data:
    nuclear = data.readlines()
    nuclear = nuclear[3].split('\t')

    exp = nuclear[5]

    nn = nuclear[13:-1:2]

cov_element = 0

nn = [float(x) for x in nn]

for n in nn:
    cov_element += n * n
cov_element /= 100

print(cov_element)