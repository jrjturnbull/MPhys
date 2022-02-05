"""
--------------------------------------------------------------------------
TO BE ADDED TO....
--------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  SymLogNorm
import sys
import math
from matplotlib.colors import LinearSegmentedColormap

root = sys.argv[1]

theory_covariance_matrix = np.load("matrices/CV_" + root + ".dat", allow_pickle=True)
correlation_matrix = np.load("matrices/CR_" + root + ".dat", allow_pickle=True)
exp_data = np.load("matrices/EXP_" + root + ".dat", allow_pickle=True)
theory_data = np.load("matrices/TH_" + root + ".dat", allow_pickle=True)
eigenvalues = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)
eigenvectors = np.load("matrices/EVC_" + root + ".dat", allow_pickle=True)
eigenvalues_norm = np.load("matrices/EVLN_" + root + ".dat", allow_pickle=True)
eigenvectors_norm = np.load("matrices/EVCN_" + root + ".dat", allow_pickle=True)
experimental_covariance_matrix = np.load("matrices/ECV_" + root + ".dat", allow_pickle=True)
pdf_covariance_matrix = np.load("matrices/XCV_" + root + ".dat", allow_pickle=True)

nuisance_parameters = np.load("matrices/NPE_" + root + ".dat", allow_pickle=True)
uncertainties_nuc = np.load("matrices/ZNE_" + root + ".dat", allow_pickle=True)
uncertainties_pdf = np.load("matrices/ZPE_" + root + ".dat", allow_pickle=True)
uncertainties_tot = np.load("matrices/ZTE_" + root + ".dat", allow_pickle=True)

autoprediction = np.load("matrices/AP_" + root + ".dat", allow_pickle=True)

autoprediction_shifts = np.load("matrices/DT_" + root + ".dat", allow_pickle=True)
theory_data_diff = np.load("matrices/TD_" + root + ".dat", allow_pickle=True)
nuclear_shifts = np.load("matrices/NSH_" + root + ".dat", allow_pickle=True)

th_contribution_1 = np.load("matrices/TH1_" + root + ".dat", allow_pickle=True)
th_contribution_2 = np.load("matrices/TH2_" + root + ".dat", allow_pickle=True)
pdf_contribution_1 = np.load("matrices/X1_" + root + ".dat", allow_pickle=True)
pdf_contribution_2 = np.load("matrices/X2_" + root + ".dat", allow_pickle=True)
pdf_contribution_3 = np.load("matrices/X3_" + root + ".dat", allow_pickle=True)

chi2_no_th = np.load("matrices/CHN_" + root + ".dat", allow_pickle=True)
chi2_yes_th = np.load("matrices/CHY_" + root + ".dat", allow_pickle=True)
chi2_shifted = np.load("matrices/CHS_" + root + ".dat", allow_pickle=True)
chi2_auto = np.load("matrices/CHA_" + root + ".dat", allow_pickle=True)

chi2_no_th_t0 = np.load("matrices/CHNt0_" + root + ".dat", allow_pickle=True)
chi2_yes_th_t0 = np.load("matrices/CHYt0_" + root + ".dat", allow_pickle=True)
chi2_shifted_t0 = np.load("matrices/CHSt0_" + root + ".dat", allow_pickle=True)

# ATTEMPT TO REPLICATE THE COLORBAR USED IN THE LITERATURE (STILL NOT QUITE RIGHT...)
c = ["maroon","firebrick","chocolate","orange","sandybrown","peachpuff","lightyellow",
        "honeydew","palegreen","aquamarine","mediumturquoise", "royalblue","midnightblue"]
v = [0,.1,.2,.3,.4,.45,.5,.55,.6,.7,.8,.9,1]
l = list(zip(v,reversed(c)))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

# LABEL REGIONS OF AXIS
def bracket(ax, pos=[0,0], scalex=1, scaley=1, text="", textkw = {}, linekw = {}):
    x = np.array([0, 0.05, 0.45, 0.5])
    y = np.array([0, -0.01, -0.01, -0.02])
    x = np.concatenate((x, x+0.5))
    y = np.concatenate((y, y[::-1]))
    ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False, transform=ax.get_xaxis_transform(), **linekw)
    ax.text(pos[0]+0.5*scalex, (y.min()-0.01)*scaley+pos[1], text, transform=ax.get_xaxis_transform(),
        ha="center", va="top", **textkw)

def show_dataset_brackets(ax):
    bracket(ax, text="1", pos=[0,0], scalex=416, scaley=3, linekw=dict(color="k", lw=2))
    bracket(ax, text="2", pos=[416,0], scalex=416, scaley=3, linekw=dict(color="k", lw=2))
    bracket(ax, text="3", pos=[832,0], scalex=85, scaley=3, linekw=dict(color="k", lw=2))
    bracket(ax, text="4", pos=[917,0], scalex=37, scaley=3, linekw=dict(color="k", lw=2))
    bracket(ax, text="5", pos=[954,0], scalex=39, scaley=3, linekw=dict(color="k", lw=2))

"""
*********************************************************************************************************
OUTPUT
_________________________________________________________________________________________________________

"""

# HEATMAP OF EXPERIMENTAL COVARIANCE MATRIX

fig, ax = plt.subplots()
experimental_covariance_matrix_norm = np.zeros_like(experimental_covariance_matrix)
for i in range(len(experimental_covariance_matrix)):
    for j in range(len(experimental_covariance_matrix)):
        experimental_covariance_matrix_norm[i,j] = experimental_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
im = ax.imshow(experimental_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-1, vmax=1), cmap=cmap)
fig.colorbar(im)
plt.title("Experimental covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
#plt.show()
plt.savefig("output/exp_covariance")
plt.clf()
plt.cla()

exp_corr = np.zeros_like(experimental_covariance_matrix)
for i in range(len(exp_corr)):
    for j in range(len(exp_corr)):
            if (experimental_covariance_matrix[i,j] == 0):
                continue
            exp_corr[i,j] = experimental_covariance_matrix[i,j] / math.sqrt(experimental_covariance_matrix[i,i] * experimental_covariance_matrix[j,j])
plt.imshow(exp_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Experimental correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/exp_correlation")
plt.clf()

# HEATMAP OF THEORY COVARIANCE MATRIX

fig, ax = plt.subplots()
theory_covariance_matrix_norm = np.zeros_like(theory_covariance_matrix)
for i in range(len(theory_covariance_matrix)):
    for j in range(len(theory_covariance_matrix)):
        theory_covariance_matrix_norm[i,j] = theory_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
im = ax.imshow(theory_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-1.5, vmax=1.5), cmap=cmap)
plt.colorbar(im)

plt.title("Theory covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
#plt.show()
plt.savefig("output/th_covariance")
plt.clf()
plt.cla()

th_corr = np.zeros_like(theory_covariance_matrix)
for i in range(len(th_corr)):
    for j in range(len(th_corr)):
            th_corr[i,j] = theory_covariance_matrix[i,j] / math.sqrt(theory_covariance_matrix[i,i] * theory_covariance_matrix[j,j])
plt.imshow(th_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Theory correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/th_correlation")
plt.clf()

# HEATMAP OF PDF UNCERTAINTIES
fig, ax = plt.subplots()
pdf_covariance_matrix_norm = np.zeros_like(pdf_covariance_matrix)
for i in range(len(pdf_covariance_matrix)):
    for j in range(len(pdf_covariance_matrix)):
        pdf_covariance_matrix_norm[i,j] = pdf_covariance_matrix[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(pdf_covariance_matrix_norm, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("PDF covariance matrix, normalised to the theory")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/pdf_covariance")
plt.clf()

pdf_correlation_matrix = np.zeros_like(pdf_covariance_matrix)
for i in range(len(pdf_covariance_matrix)):
    for j in range(len(pdf_covariance_matrix)):
        pdf_correlation_matrix[i,j] = pdf_covariance_matrix[i,j] / math.sqrt(pdf_covariance_matrix[i,i] * pdf_covariance_matrix[j,j])
plt.imshow(pdf_correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar()
plt.title("Heatmap of PDF correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/pdf_correlation")
plt.clf()


# PLOT OF DIAGONAL ELEMENTS NORMALISED TO THE THEORY
C_norm = np.array([math.sqrt(experimental_covariance_matrix_norm[i,i]) for i in range(len(experimental_covariance_matrix_norm))])
S_norm = np.array([math.sqrt(theory_covariance_matrix_norm[i,i]) for i in range(len(theory_covariance_matrix_norm))])
X_norm = np.array([math.sqrt(pdf_covariance_matrix_norm[i,i]) for i in range(len(pdf_covariance_matrix_norm))])

x = np.arange(len(C_norm))
plt.scatter(x, X_norm, c='orange', s=1.5, label='X')
plt.scatter(x, C_norm, c='g', s=1.5, label='C')
plt.scatter(x, S_norm, c='purple', s=1.5, label='S')
plt.title("Sqaure root of diagonal elements, normalised to the theory")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("output/diagonal_elements.png")
plt.clf()

# NON-ZERO EIGENVALUES
nz_eigen = [i for i in range(len(eigenvalues)) if eigenvalues[i] > 1e-5]
eigenvalues_nz = np.array([eigenvalues[i] for i in nz_eigen])[::-1]

x = np.arange(len(eigenvalues_nz))
plt.scatter(x, eigenvalues_nz, s=1.5)
plt.title("Non-zero eigenvalues of S (cutoff = 1e-5)")
plt.yscale('log')
plt.gca().axes.xaxis.set_visible(False)
#plt.show()
plt.savefig("output/nz_eigenvalues.png")
plt.clf()

nz_eigen_norm = [i for i in range(len(eigenvalues_norm)) if eigenvalues_norm[i] > 1e-5]
eigenvalues_nz_norm = np.array([eigenvalues_norm[i] for i in nz_eigen_norm])[::-1]

x = np.arange(len(eigenvalues_nz_norm))
plt.scatter(x, eigenvalues_nz_norm, s=1.5)
plt.title("Non-zero eigenvalues of S (cutoff = 1e-5), normalised to the theory")
plt.yscale('log')
plt.gca().axes.xaxis.set_visible(False)
#plt.show()
plt.savefig("output/nz_eigenvalues_norm.png")
plt.clf()

# AUTOPREDICTION MATRICES
autoprediction_norm = np.zeros_like(autoprediction)
for i in range(len(autoprediction)):
    for j in range(len(autoprediction)):
        autoprediction_norm[i,j] = autoprediction[i,j] / (theory_data[i] * theory_data[j])
plt.imshow(autoprediction_norm, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
plt.colorbar()
plt.title("Autoprediction covariance matrix, normalised to the theory")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/autoprediction_covariance")
plt.clf()

autoprediction_corr = np.zeros_like(autoprediction)
for i in range(len(autoprediction)):
    for j in range(len(autoprediction)):
            autoprediction_corr[i,j] = autoprediction[i,j] / math.sqrt(autoprediction[i,i] * autoprediction[j,j])
plt.imshow(autoprediction_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Autoprediction correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("output/autoprediction_correlation")
plt.clf()

autoprediction_cons = pdf_covariance_matrix + theory_covariance_matrix

unc_ap = np.array([math.sqrt(autoprediction[i,i])/ theory_data[i] for i in range(len(autoprediction))])
unc_pdf = np.array([math.sqrt(pdf_covariance_matrix[i,i]) / theory_data[i] for i in range(len(pdf_covariance_matrix))])
unc_cons = np.array([math.sqrt(autoprediction_cons[i,i]) / theory_data[i] for i in range(len(autoprediction_cons))])

x = np.arange(len(unc_ap))
plt.scatter(x, unc_ap, c='b', s=1, label='P')
plt.scatter(x, unc_cons, c='cyan', s=1, label='P_cons')
plt.scatter(x, unc_pdf, c='r', s=1, label='X')
plt.title("Autoprediction percentage uncertainties")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("output/autoprediction_uncertainties")
plt.clf()


# AUTOPREDICTION SHIFTS
autoprediction_shifts_norm = np.zeros_like(autoprediction_shifts)
for i in range(len(autoprediction_shifts)):
    autoprediction_shifts_norm[i] = autoprediction_shifts[i] / theory_data[i]
theory_data_diff_norm = np.zeros_like(theory_data_diff)
for i in range(len(theory_data_diff)):
    theory_data_diff_norm[i] = theory_data_diff[i] / theory_data[i]
nuclear_shifts_norm = np.zeros_like(nuclear_shifts)
for i in range(len(nuclear_shifts)):
    nuclear_shifts_norm[i] = nuclear_shifts[i] / theory_data[i]

x = np.arange(len(autoprediction_shifts))
plt.ylim(-1.4,1.4)
plt.plot(x, -theory_data_diff_norm, c='cyan', label='D-T', linewidth=0.35, zorder=2)
plt.plot(x, autoprediction_shifts_norm, c='b', label='δT', linewidth=0.35, zorder=2)
plt.plot(x, nuclear_shifts_norm, c='r', label='Nuclear Shifts', linewidth=0.35, zorder=2)
plt.title("Autoprediction shifts compared to theory-data differences")
show_dataset_brackets(plt.gca())
plt.axhline(y=0, color='k', linestyle='-', zorder=1)
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("output/autoprediction_shifts")
plt.clf()


# CONTRIBUTIONS TO THE DIAGONAL ELEMENTS OF THE CORRELATED THEORY & PDF UNCERTAINTIES
th_contribution_1_norm = np.zeros(shape=len(th_contribution_1))
for i in range(len(th_contribution_1)):
    th_contribution_1_norm[i] = th_contribution_1[i,i] / theory_covariance_matrix[i,i]
th_contribution_2_norm = np.zeros(shape=len(th_contribution_2))
for i in range(len(th_contribution_2)):
    th_contribution_2_norm[i] = th_contribution_2[i,i] / theory_covariance_matrix[i,i]

x = np.arange(len(th_contribution_1_norm))
plt.scatter(x, th_contribution_1_norm, c='fuchsia', s=1.5, label=r'$S-S(C+S)^{-1}S$')
plt.scatter(x, th_contribution_2_norm, c='black', s=1.5, label=r'$S-S(C+S)^{-1}S + S(C+S)^{-1}X(C+S)^{-1}S$')
plt.title("Diagonal contributions to the theory uncertainties")
show_dataset_brackets(plt.gca())
plt.axhline(y=0, color='k', linestyle='-')
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("output/theory_contributions")
plt.clf()

pdf_contribution_1_norm = np.zeros(shape=len(pdf_contribution_1))
for i in range(len(pdf_contribution_1)):
    pdf_contribution_1_norm[i] = pdf_contribution_1[i,i] / pdf_covariance_matrix[i,i]
pdf_contribution_2_norm = np.zeros(shape=len(pdf_contribution_2))
for i in range(len(pdf_contribution_2)):
    pdf_contribution_2_norm[i] = pdf_contribution_2[i,i] / pdf_covariance_matrix[i,i]
pdf_contribution_3_norm = np.zeros(shape=len(pdf_contribution_3))
for i in range(len(pdf_contribution_3)):
    pdf_contribution_3_norm[i] = pdf_contribution_3[i,i] / pdf_covariance_matrix[i,i]

x = np.arange(len(pdf_contribution_1_norm))
plt.scatter(x, pdf_contribution_1_norm, c='lightgreen', s=1.5, label=r'$C(C+S)^{-1}X(C+S)^{-1}C$')
plt.scatter(x, pdf_contribution_2_norm, c='pink', s=1.5, label=r'$X-S(C+S)^{-1}X - X(C+S)^{-1}S$')
#plt.scatter(x, pdf_contribution_2_norm + pdf_contribution_3_norm, c='k', s=1.5, label='NEW TERM')
plt.title("Diagonal contributions to the PDF uncertainties")
show_dataset_brackets(plt.gca())
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0, color='k', linestyle='-')
#plt.ylim(-1.6, 1.1)
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("output/pdf_contributions")
plt.clf()


# NUISANCE PARAMETER EXPECTATION VALUES
x = np.arange(len(nuisance_parameters))
plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, nuisance_parameters, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_nuc, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=6)
plt.title("Nuisance parameters with nuclear uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("output/NPE_nuc")
plt.clf()

plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, nuisance_parameters, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_pdf, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=7)
plt.title("Nuisance parameters with PDF uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("output/NPE_pdf")
plt.clf()

plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, nuisance_parameters, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,nuisance_parameters,yerr=uncertainties_tot, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=7)
plt.title("Nuisance parameters with total uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("output/NPE_tot")
plt.clf()

# CHI2 PLOT (manual data insertion)
x1 = np.array([0.8,1.8,2.8,3.8,4.8])
x2 = np.array([1,2,3,4,5])
x3 = np.array([1.2,2.2,3.2,4.2,5.2])
x4 = np.array([1.4,2.4,3.4,4.4,5.4])
plt.bar(x1, height=chi2_no_th, width=0.15, color='blue', label='chi2_no_th')
plt.bar(x2, height=chi2_yes_th, width=0.15, color='orange', label='chi2_yes_th')
plt.bar(x3, height=chi2_shifted, width=0.15, color='red', label='chi2_shifted')
plt.bar(x4, height=chi2_auto, width=0.15, color='green', label='chi2_auto')
plt.legend()
labels = [item.get_text() for item in plt.gca().get_xticklabels()]
labels = ['', 'CHORUS_nb', 'CHORUS_nu', 'DYE605', 'NTV_nb', 'NTV_nu']
plt.gca().set_xticklabels(labels)
plt.title("Chi squared for the various processes")
plt.savefig("output/chi2")
plt.clf()

# CHI2 PLOT (manual data insertion)
x1 = np.array([0.8,1.8,2.8,3.8,4.8])
x2 = np.array([1,2,3,4,5])
x3 = np.array([1.2,2.2,3.2,4.2,5.2])
x4 = np.array([1.4,2.4,3.4,4.4,5.4])
plt.bar(x1, height=chi2_no_th_t0, width=0.15, color='blue', label='chi2_no_th')
plt.bar(x2, height=chi2_yes_th_t0, width=0.15, color='orange', label='chi2_yes_th')
plt.bar(x3, height=chi2_shifted_t0, width=0.15, color='red', label='chi2_shifted')
#plt.bar(x4, height=chi2_auto_t0, width=0.15, color='green', label='chi2_auto')
plt.legend()
labels = [item.get_text() for item in plt.gca().get_xticklabels()]
labels = ['', 'CHORUS_nb', 'CHORUS_nu', 'DYE605', 'NTV_nb', 'NTV_nu']
plt.gca().set_xticklabels(labels)
plt.title("Chi squared for the various processes (t0 method)")
plt.savefig("output/chi2t0")
plt.clf()