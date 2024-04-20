# -*- coding: utf-8 -*-
"""
Plot of experimental data with fit curve of a parameter obtained with
Monte-Carlo simulations.
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors, cm, colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = \
    r"\usepackage{siunitx}"
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

DPI = 300 # dots per inch

mm = 1/25.4 # To set the size of the figure in mm rather than inch


########## Colors in plots ##########
def tinting(color: np.ndarray, tint: float)-> np.ndarray:
    """
    Tint a color (make it lighter). 0 <= tint <= 1
    tint = 1 -> white, tint = 0 -> color unchanged
    """
    c = np.copy(color)
    c[..., :3] = tint + (1-tint) * color[..., :3]
    return c

ms = 2.5 # marker size
mew = 0.6 # marker edge width
elw = 0.5 # error linewidth
## colors
ncolors = np.empty((6, 4, 4), dtype=float)
# raw colors
ncolors[0, 0, :] = colors.to_rgba("royalblue")
ncolors[1, 0, :] = colors.to_rgba("coral")
ncolors[2, 0, :] = colors.to_rgba("tab:red")
ncolors[3, 0, :] = colors.to_rgba("tab:green")
ncolors[4, 0, :] = colors.to_rgba("tab:red")
ncolors[5, 0, :] = colors.to_rgba("yellowgreen")
tint = 0.4
# marker edge colors
ncolors[:, 1, :] = ncolors[:, 0, :]
ncolors[:, 1, -1] = 1 # set alpha
# marker face color
ncolors[:, 2, :] = tinting(ncolors[:, 0, :], tint)
ncolors[:, 2, -1] = 1 # set alpha
# error bar color
ncolors[:, 3, :] = ncolors[:, 1, :]
# 
rawc = ncolors[:, 0, :]
mec = ncolors[:, 1, :]
fc = ncolors[:, 2, :]
ec = ncolors[:, 3, :]


########## Parameters for text and arrows ##########
lfs = 9 # label fontsize
tlfs = 7 # ticklabel fontsize


########### Experimental data stuff ##########
# Load data
sim_file = "./simulations.npz"
fit_data_file = "./fit_data.npz"
with np.load(sim_file, allow_pickle=False) as f:
    xval = f['xvalues']
    precap = f['precap'][0]
    stdev_precap = f['stdev_precap'][0]
    sim_temp = f['temperatures']*1e6
    times = f['delays'] * 1e6
    simu = f['simulations']
with np.load(fit_data_file, allow_pickle=False) as f:
    coarse_temperatures = f['coarse_temperatures']*1e6
    coarse_chi2 = f['coarse_chisq']*1e-3
    precise_temperatures = f['precise_temperatures']*1e6
    precise_chi2 = f['precise_chisq']*1e-3
    Tfit = f['best_T']*1e6
    polyfit = f['polyfit']
    best_chi2 = f['best_chisq']


# Build custom colormap
vmin, vmax = 0., 2.
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = colormaps.get_cmap("plasma")
cmap = colors.ListedColormap(cmap(np.linspace(0.0, 1., 256)))

# Eval polynomial fit 
domain = (np.min(precise_temperatures), np.max(precise_temperatures))
x = np.linspace(9, 21, 121, endpoint=True)
poly = Polynomial(polyfit, domain=domain)


# =============================================================================
# PLOT - FIGURE INITIALIZATION
# =============================================================================

## Initialize the figure
fig = plt.figure(
    num=0,
    figsize=(152*mm, 55*mm),
    dpi=DPI,
    clear=True,
    facecolor='white')


gs = gridspec.GridSpec(
    nrows=1, ncols=2,
    height_ratios=[1.], hspace=0,
    width_ratios=[0.9, 1.], wspace=0.48,
    figure=fig,
    left=0.07, bottom=(b:=0.15), right=(r:=0.99), top=(t:=0.96))

fig.text(0.02, 0.96, r"\textbf{(a)}", fontsize=11, ha="center", va="center")
fig.text(0.525, 0.96, r"\textbf{(b)}", fontsize=11, ha="center", va="center")

# =============================================================================
# SIMUlATIONS
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])

ax1.set_xlim(0, 160)
ax1.set_ylim(5e-2, 1)
ax1.set_yscale("log", nonpositive='clip')

ax1.tick_params(
    axis='both', direction='out',
    labelsize=tlfs, pad=1.5,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax1.set_xticks([0, 50, 100, 150])
ax1.set_xticks([25, 75, 125], minor=True)
ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 1.],
                labels=["$0.1$", "$0.2$", "$0.3$", "$0.4$", "$0.5$", "$1$"])
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)


ax1.errorbar(
    xval, precap, yerr=stdev_precap,
    fmt='o', markersize=ms, mfc=fc[0], mec=mec[0], mew=mew,
    elinewidth=elw, ecolor=ec[0],)
for i, T in enumerate(sim_temp):
    ax1.plot(times, simu[i],
             marker="", ls="-", color=cmap(np.log10(T)/2), lw=0.8,
             alpha=0.9)

ax1.set_xlabel(r"Release duration ($\si{\us}$)", fontsize=lfs, labelpad=2)
ax1.set_ylabel(r"Recapture prob.", fontsize=lfs, labelpad=4)


########## Colorbar ##########
x0, y0, dx, dy = ax1.get_position().bounds
x1 = x0 + dx
cax1 = fig.add_axes(rect=(x1+0.01, y0, 0.02, dy))
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1,
                  orientation="vertical",
                  ticklocation="right", alpha=0.9)
cax1.tick_params(axis="y", which="major", direction="in",
                 size=3.,
                 color="0.", labelsize=tlfs, pad=2)
cax1.tick_params(axis="y", which="minor", direction="in",
                 size=2.,
                 color="0.", labelsize=tlfs, pad=1)

cax1.set_yticks([0., 1., 2.], [r"$1$", r"$10$", r"$100$"], minor=False)
cax1.set_yticks(np.log10([2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90]),
                minor=True)
cax1.set_ylabel(r"Temperature ($\si{\micro\K}$)",
                ha="center", va="center", rotation=-90, labelpad=2, fontsize=lfs)


# =============================================================================
# CHI2
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

### Average recapture probability and simulations
ax2.set_xlim((0, 80))
ax2.set_ylim(0., 1000)
ax2.tick_params(
    axis='both', direction='out', labelsize=tlfs, pad=2,
    top=False, labeltop=False,
    bottom=True, labelbottom=True,
    left=True, labelleft=True,
    right=False, labelright=False
    )
ax2.set_xticks([0, 20, 40, 60, 80])
ax2.set_yticks([0., 200, 400, 600, 800, 1000])
ax2.grid(which='minor', alpha=0.2)
ax2.grid(which='major', alpha=0.5)

l_chi2, = ax2.plot(
    coarse_temperatures, coarse_chi2,
    marker='o', markersize=ms, mfc=fc[0], mec=mec[0], mew=mew,
    ls="")
l_poly, = ax2.plot(
    T:=np.linspace(3.55, 40., 81), poly(T)*1e-3,
    marker="", ls="-", color=rawc[2], lw=0.6)
ax2.plot(
    T:=np.linspace(0., 3.1, 32), poly(T)*1e-3,
    marker="", ls="-", color=rawc[2], lw=0.6)
l_Tfit, = ax2.plot(
    [Tfit, Tfit], ax2.get_ylim(),
         marker="", ls="--", color=rawc[3], lw=0.8)

ax2.set_xlabel(r"Temperature ($\si{\micro\K}$)", fontsize=lfs, labelpad=2)
ax2.set_ylabel(r"$\chi^2 \ (\times 10^3)$", fontsize=lfs, labelpad=1)

########## Legend ##########
ax2.legend(
    handles=[l_chi2, l_poly, l_Tfit],
    labels=[r"computed $\chi^2$", r"polynomial fit", r"$\hat{T} = \SI{14.1}{\micro\K}$"],
    bbox_to_anchor=(0.55, 0., 0.45, 0.4), loc="lower right",
    facecolor="w", edgecolor="0.8", framealpha=1.0, ncol=1,
    mode="expand", borderaxespad=0.2, prop={"size": 7},
    borderpad=0.4, labelspacing=0.5, handletextpad=0.6,
    handleheight=0.2, handlelength=1.6)


########## Inset: Zoom on the minimum ##########
x0, x1, y0, y1 = 10, 20, 0., 40
rect = patches.Rectangle(
    (x0, y0), x1-x0, y1-y0, facecolor="none", edgecolor="0.4",
    linewidth=0.6, zorder=3)
ax2.add_patch(rect)

ax2ins = inset_axes(
    ax1, width="100%", height="100%", loc="upper left", borderpad=0.2,
    bbox_to_anchor=(0.025, 0.56, 0.49, 0.43), bbox_transform=ax2.transAxes)
ax2ins.tick_params(axis='x', which="both", direction="in",
                    labelsize=6, pad=2,
                    bottom=True, labelbottom=True,
                    top=True, labeltop=False)
ax2ins.tick_params(axis='y', which="both", direction="in",
                    labelsize=6, pad=2,
                    left=True, labelleft=False,
                    right=True, labelright=True)
ax2ins.set_xlim(x0, x1)
ax2ins.set_ylim(y0, y1)
ax2ins.set_xticks([10, 15, 20], minor=False)
ax2ins.set_xticks([12.5, 17.5], minor=True)
ax2ins.set_yticks([0, 20, 40], minor=False)
ax2ins.set_yticks([10, 30], minor=True)
ax2ins.grid(which='minor', alpha=0.2)
ax2ins.grid(which='major', alpha=0.5)

ax2ins.plot(precise_temperatures, precise_chi2,
            marker='o', markersize=ms, mfc=fc[0], mec=mec[0], mew=mew,
            ls="")
ax2ins.plot(x, poly(x)*1e-3,
            marker='', ls="-", color=rawc[2], lw=0.8)
ax2ins.plot([Tfit, Tfit], ax2ins.get_ylim(),
         marker="", ls="--", color=rawc[3], lw=0.8)


################################# SAVE FIGURE #################################
fig.savefig("./data_plot.png")
