import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  SymLogNorm
import sys
import math
from matplotlib.colors import LinearSegmentedColormap

root = sys.argv[1]

C = np.load("matrices/C_" + root + ".dat", allow_pickle=True)
S = np.load("matrices/S_" + root + ".dat", allow_pickle=True)
X = np.load("matrices/X_" + root + ".dat", allow_pickle=True)
P = np.load("matrices/P_" + root + ".dat", allow_pickle=True)

T = np.load("matrices/T_" + root + ".dat", allow_pickle=True)
D = np.load("matrices/D_" + root + ".dat", allow_pickle=True)
T_KFAC = np.load("matrices/TK_" + root + ".dat", allow_pickle=True)

NPE = np.load("matrices/NPE_" + root + ".dat", allow_pickle=True)
NPE_nuc = np.load("matrices/NPEnuc_" + root + ".dat", allow_pickle=True)
NPE_pdf = np.load("matrices/NPEpdf_" + root + ".dat", allow_pickle=True)
NPE_tot = np.load("matrices/NPEtot_" + root + ".dat", allow_pickle=True)

AUTO = np.load("matrices/AUTO_" + root + ".dat", allow_pickle=True)

S1 = np.load("matrices/S1_" + root + ".dat", allow_pickle=True)
S2 = np.load("matrices/S2_" + root + ".dat", allow_pickle=True)
X1 = np.load("matrices/S1_" + root + ".dat", allow_pickle=True)
X2 = np.load("matrices/S2_" + root + ".dat", allow_pickle=True)

EVAL = np.load("matrices/EVL_" + root + ".dat", allow_pickle=True)

#CHI2 = np.load("matrices/CHI2_" + root + ".dat", allow_pickle=True)
#CHI2t0 = np.load("matrices/CHI2t0_" + root + ".dat", allow_pickle=True)

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

    if(root == "nuclear" or root == 'nuclear30'):
        bracket(ax, text="1", pos=[0,0], scalex=416, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="2", pos=[416,0], scalex=416, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="3", pos=[832,0], scalex=85, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="4", pos=[917,0], scalex=37, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="5", pos=[954,0], scalex=39, scaley=3, linekw=dict(color="k", lw=2))
    elif(root == "deuterium" or root == "deuterium30"):
        bracket(ax, text="1", pos=[0,0], scalex=248, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="2", pos=[248,0], scalex=15, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="3", pos=[263,0], scalex=121, scaley=3, linekw=dict(color="k", lw=2))
        bracket(ax, text="4", pos=[385,0], scalex=34, scaley=3, linekw=dict(color="k", lw=2))
    else:
        print("Unable to show dataset brackets: root not recognised...")


#######################################################################

#region covariance/correlation matrices

# C matrix
fig, ax = plt.subplots()
C_norm = np.zeros_like(C)
for i in range(len(C)):
    for j in range(len(C)):
        C_norm[i,j] = C[i,j] / (T[i] * T[j])
im = ax.imshow(C_norm, norm=SymLogNorm(1e-4,vmin=-1, vmax=1), cmap=cmap)
fig.colorbar(im)
plt.title("Experimental covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
plt.savefig("figures/" + root + "/C_covariance")
plt.clf()
plt.cla()

C_corr = np.zeros_like(C)
for i in range(len(C)):
    for j in range(len(C)):
            if (C[i,j] == 0):
                continue
            C_corr[i,j] = C[i,j] / math.sqrt(C[i,i] * C[j,j])
plt.imshow(C_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Experimental correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("figures/" + root + "/C_correlation")
plt.clf()

# S matrix
fig, ax = plt.subplots()
S_norm = np.zeros_like(S)
for i in range(len(S)):
    for j in range(len(S)):
        S_norm[i,j] = S[i,j] / (T[i] * T[j])
im = ax.imshow(S_norm, norm=SymLogNorm(1e-4,vmin=-1.5, vmax=1.5), cmap=cmap)
fig.colorbar(im)
plt.title("Theory covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
plt.savefig("figures/" + root + "/S_covariance")
plt.clf()
plt.cla()

S_corr = np.zeros_like(S)
for i in range(len(S)):
    for j in range(len(S)):
            if (S[i,j] == 0):
                continue
            S_corr[i,j] = S[i,j] / math.sqrt(S[i,i] * S[j,j])
plt.imshow(S_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Theory correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("figures/" + root + "/S_correlation")
plt.clf()

# X matrix
fig, ax = plt.subplots()
X_norm = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(X)):
        X_norm[i,j] = X[i,j] / (T[i] * T[j])
im = ax.imshow(X_norm, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
fig.colorbar(im)
plt.title("PDF covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
plt.savefig("figures/" + root + "/X_covariance")
plt.clf()
plt.cla()

X_corr = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(X)):
            if (X[i,j] == 0):
                continue
            X_corr[i,j] = X[i,j] / math.sqrt(X[i,i] * X[j,j])
plt.imshow(X_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("PDF correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("figures/" + root + "/X_correlation")
plt.clf()

# P matrix
fig, ax = plt.subplots()
P_norm = np.zeros_like(P)
for i in range(len(P)):
    for j in range(len(P)):
        P_norm[i,j] = P[i,j] / (T[i] * T[j])
im = ax.imshow(X_norm, norm=SymLogNorm(1e-4,vmin=-0.1, vmax=0.1), cmap=cmap)
fig.colorbar(im)
plt.title("Autoprediction covariance matrix, normalised to the theory")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
show_dataset_brackets(ax)
plt.savefig("figures/" + root + "/P_covariance")
plt.clf()
plt.cla()

P_corr = np.zeros_like(P)
for i in range(len(P)):
    for j in range(len(P)):
            if (P[i,j] == 0):
                continue
            P_corr[i,j] = P[i,j] / math.sqrt(P[i,i] * P[j,j])
plt.imshow(P_corr ,vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
plt.title("Autoprediction correlation matrix")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
#plt.show()
plt.savefig("figures/" + root + "/P_correlation")
plt.clf()

# Diagonal contributions
X_diag = np.array([math.sqrt(X[i,i])/T[i] for i in range(len(X))])
C_diag = np.array([math.sqrt(C[i,i])/T[i] for i in range(len(X))])
S_diag = np.array([math.sqrt(S[i,i])/T[i] for i in range(len(X))])
x = np.arange(len(X_diag))

plt.scatter(x, X_diag, c='orange', s=1.5, label='X')
plt.scatter(x, C_diag, c='green', s=1.5, label='C')
plt.scatter(x, S_diag, c='purple', s=1.5, label='S')
plt.title("Square root of diagonal elements, normalised to the theory")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
plt.savefig("figures/" + root + "/diagonal_elements")
plt.clf()

#endregion

#region autoprediction shifts
AUTO_norm = np.zeros_like(AUTO)
for i in range(len(AUTO)):
    AUTO_norm[i] = AUTO[i] / T[i]
TD_norm = np.zeros_like(T-D)
for i in range(len(TD_norm)):
    TD_norm[i] = (T-D)[i] / T[i]
TK_norm = np.zeros_like(T-T_KFAC)
for i in range(len(TK_norm)):
    TK_norm[i] = (T-T_KFAC)[i] / T[i]

x = np.arange(len(AUTO))
#plt.ylim(-1.4,1.4)
plt.plot(x, -TD_norm, c='cyan', label='D-T', linewidth=0.35, zorder=2)
plt.plot(x, AUTO_norm, c='b', label='δT', linewidth=0.35, zorder=2)
plt.plot(x, -TK_norm, c='r', label='Nuclear shifts', linewidth=0.35, zorder=2)
plt.title("Autoprediction shifts compared to theory-data differences")
show_dataset_brackets(plt.gca())
plt.axhline(y=0, color='k', linestyle='-', zorder=1)
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("figures/" + root + "/autopredictions")
plt.clf()

P_uncert = np.array([math.sqrt(P[i,i])/T[i] for i in range(len(P))])
X_uncert = np.array([math.sqrt(X[i,i])/T[i] for i in range(len(X))])
Pc_uncert = np.array([math.sqrt((X+S)[i,i])/T[i] for i in range(len(X))])
x = np.arange(len(P))
plt.scatter(x, P_uncert, c='b', s=1, label=r'$P$')
plt.scatter(x, Pc_uncert, c='cyan', s=1, label=r'$P^{con}$')
plt.scatter(x, X_uncert, c='r', s=1, label=r'$X$')
plt.title("Percentage uncertainties")
show_dataset_brackets(plt.gca())
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("figures/" + root + "/uncertainties")
plt.clf()

#endregion

#region eigenvalues + nuisance parameters
nz_eigen = [i for i in range(len(EVAL)) if EVAL[i] > 1e-5]
eigenvalues_nz = np.array([EVAL[i] for i in nz_eigen])[::-1]
x = np.arange(len(eigenvalues_nz))
plt.scatter(x, eigenvalues_nz, s=1.5)
plt.title("Non-zero eigenvalues of S (cutoff = 1e-5)")
plt.yscale('log')
plt.gca().axes.xaxis.set_visible(False)
#plt.show()
plt.savefig("figures/" + root + "/eigenvalues")
plt.clf()

x = np.arange(len(NPE))
plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, NPE, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,NPE,yerr=NPE_nuc, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=6)
plt.title("Nuisance parameters with nuclear uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("figures/" + root + "/NPE_nuc")
plt.clf()

plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, NPE, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,NPE,yerr=NPE_pdf, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=7)
plt.title("Nuisance parameters with PDF uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("figures/" + root + "/NPE_pdf")
plt.clf()

plt.ylim(-4.2,4.2)
plt.axhspan(-1, 1, color='yellow', alpha=0.5, zorder=1)
plt.scatter(x, NPE, vmin=-2, vmax=2, zorder=5)
plt.errorbar(x,NPE,yerr=NPE_tot, ls='none', zorder=6)
plt.axhline(y=0, color='k', linestyle='-', zorder=7)
plt.title("Nuisance parameters with total uncertainties")
plt.gca().axes.xaxis.set_visible(False)
plt.savefig("figures/" + root + "/NPE_tot")
plt.clf()
#endregion

#region diagonal contributions
S1_norm = np.zeros(shape=len(S1))
for i in range(len(S1)):
    S1_norm[i] = S1[i,i] / S[i,i]
S2_norm = np.zeros(shape=len(S2))
for i in range(len(S2)):
    S2_norm[i] = S2[i,i] / S[i,i]

x = np.arange(len(S1_norm))
plt.scatter(x, S1_norm, c='fuchsia', s=1.5, label=r'$S-S(C+S)^{-1}S$')
plt.scatter(x, S2_norm, c='black', s=1.5, label=r'$S-S(C+S)^{-1}S + S(C+S)^{-1}X(C+S)^{-1}S$')
plt.title("Diagonal contributions to the theory uncertainties")
show_dataset_brackets(plt.gca())
plt.axhline(y=0, color='k', linestyle='-')
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("figures/" + root + "/S_contributions")
plt.clf()

X1_norm = np.zeros(shape=len(X1))
for i in range(len(X1)):
    X1_norm[i] = X1[i,i] / X[i,i]
X2_norm = np.zeros(shape=len(X2))
for i in range(len(X2)):
    X2_norm[i] = X2[i,i] / X[i,i]

x = np.arange(len(X1_norm))
plt.scatter(x, X1_norm, c='lightgreen', s=1.5, label=r'$C(C+S)^{-1}X(C+S)^{-1}C$')
plt.scatter(x, X2_norm, c='pink', s=1.5, label=r'$X-S(C+S)^{-1}X - X(C+S)^{-1}S$')
#plt.scatter(x, pdf_contribution_2_norm + pdf_contribution_3_norm, c='k', s=1.5, label='NEW TERM')
plt.title("Diagonal contributions to the PDF uncertainties")
show_dataset_brackets(plt.gca())
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0, color='k', linestyle='-')
#plt.ylim(-1.6, 1.1)
plt.gca().axes.xaxis.set_visible(False)
plt.legend()
#plt.show()
plt.savefig("figures/" + root + "/X_contributions")
plt.clf()
#endregion

"""
#region chi2

if (root == "nuclear"):
    x1 = np.array([0.8,1.8,2.8,3.8,4.8])
    x2 = np.array([1,2,3,4,5])
    x3 = np.array([1.2,2.2,3.2,4.2,5.2])
elif (root == "deuterium" or root == "nuclear30):
    x1 = np.array([0.8,1.8,2.8,3.8])
    x2 = np.array([1,2,3,4])
    x3 = np.array([1.2,2.2,3.2,4.2])
else:
    print("Error: root not recognised")

# exp method
plt.ylim(0,1.4)
plt.bar(x1, height=CHI2[0], width=0.15, color='blue', label='nonuclear')
plt.bar(x2, height=CHI2[1], width=0.15, color='orange', label='noshift')
plt.bar(x3, height=CHI2[2], width=0.15, color='red', label='shift')

if (root == "nuclear"):
    plt.gca().set_xticklabels(('', 'CHORUS_nb', 'CHORUS_nu', 'DYE605', 'NTV_nb', 'NTV_nu'))
elif (root == "deuterium"):
    plt.gca().set_xticklabels(('', 'BCDMSD','', 'DYE886R','', 'NMCPD','', 'SLACD')) # extra spaces because it works somehow...
else:
    print("Error: root not recognised")

plt.legend()
plt.title("Chi squared for the various processes, exp method")
plt.savefig("figures/" + root + "/chi2")
plt.clf()

# t0 method
plt.ylim(0,1.4)
plt.bar(x1, height=CHI2t0[0], width=0.15, color='blue', label='nonuclear')
plt.bar(x2, height=CHI2t0[1], width=0.15, color='orange', label='noshift')
plt.bar(x3, height=CHI2t0[2], width=0.15, color='red', label='shift')

if (root == "nuclear"):
    plt.gca().set_xticklabels(('', 'CHORUS_nb', 'CHORUS_nu', 'DYE605', 'NTV_nb', 'NTV_nu'))
elif (root == "deuterium"):
    plt.gca().set_xticklabels(('', 'BCDMSD','', 'DYE886R','', 'NMCPD','', 'SLACD')) # extra spaces because it works somehow...
else:
    print("Error: root not recognised")

plt.legend()
plt.title("Chi squared for the various processes, t0 method")
plt.savefig("figures/" + root + "/chi2t0")
plt.clf()

#endregion
"""