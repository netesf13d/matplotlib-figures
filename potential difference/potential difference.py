# -*- coding: utf-8 -*-
"""
Cut views of the difference between two trapping potentials for atoms with
isocontour lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
from matplotlib import colormaps

from scipy.constants import physical_constants
c = physical_constants['speed of light in vacuum'][0]
h = physical_constants['Planck constant'][0]
e = physical_constants['elementary charge'][0]
alpha = physical_constants['fine-structure constant'][0]
m_e = physical_constants['electron mass'][0]
lambda_trap = 0.821 * 1e-6 # m
omega_trap = 2*np.pi * c / lambda_trap # rad/s
beta_ponder = alpha /(m_e * omega_trap**2) # Ponderomotive strength
P0 = 1. # W


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = \
    r"\usepackage{siunitx}"
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

DPI = 300 # dots per inch

mm = 1/25.4 # To set the size of the figure in mm rather than inch


########## Parameters for text and arrows ##########
lfs = 9 # label fontsize
tlfs = 7 # ticklabel fontsize
tcolor = "0." # ticks color


########### Experimental data stuff ##########
## Load data
sect_file = "./potential_difference_data.npz"
with np.load(sect_file, allow_pickle=False) as f:
    extent = f['extent'] * 1e6
    sh = f['shape']
    df = (f['Emin52'] - f['Emin50']) * beta_ponder * P0 / 1e3
    
    bob_xy = (f['bob52_sect_xy'] - f['bob50_sect_xy']) * beta_ponder * P0 / 1e3
    bob_xz = (f['bob52_sect_xz'] - f['bob50_sect_xz']) * beta_ponder * P0 / 1e3
    bob_yz = (f['bob52_sect_yz'] - f['bob50_sect_yz']) * beta_ponder * P0 / 1e3
    
    edge_xy = f['bob52_edge_xy']
    edge_xz = f['bob52_edge_xz']
    edge_yz = f['bob52_edge_yz']
    
XX = [np.linspace(*extent[0], sh[0], endpoint=True),
      np.linspace(*extent[1], sh[1], endpoint=True),
      np.linspace(*extent[2], sh[2], endpoint=True)]


bmin, bmax = np.min(bob_xy), np.max(bob_xy)

bob_xy[np.nonzero(edge_xy)] = np.nan
bob_xz[np.nonzero(edge_xz)] = np.nan
bob_yz[np.nonzero(edge_yz)] = np.nan

ext = [extent[[0, 0, 1, 1], [0, 1, 0, 1]],
       extent[[2, 2, 0, 0], [0, 1, 0, 1]],
       extent[[2, 2, 1, 1], [0, 1, 0, 1]]]
width_z, width_x = ext[2][1], ext[0][1]
bob = [bob_xy, bob_xz, bob_yz]



### Build custom colormap
vmin, vmax = bmin, bmax
cyan = 0.36
yellow = 0.65
cmin, cmax = 0., 0.92
fscale = vmax - vmin
x1, x2, x3 = abs(vmin)/fscale, df/fscale, (bmax-df)/fscale
n1, n2, n3 = int(x1 * 256), int(x2 * 256), int(x3 * 256)

cspace = np.concatenate((np.linspace(cmin, cyan, n1, endpoint=False),
                         np.linspace(cyan, yellow, n2, endpoint=False),
                         np.linspace(yellow, cmax, n3+1, endpoint=True)))

rcmap = colormaps.get_cmap("jet")
cmap = rcmap(cspace)
cmap = colors.ListedColormap(cmap)
cmap.set_bad("0.5")

norm = colors.Normalize(vmin=vmin, vmax=vmax)


# =============================================================================
# PLOT - FIGURE INITIALIZATION
# =============================================================================

## Initialize the figure
fig = plt.figure(
    num=0,
    figsize=(152*mm, 70*mm),
    dpi=DPI,
    clear=True,
    facecolor='white')


# =============================================================================
# XY CUT
# =============================================================================

ax0 = fig.add_axes(rect=((l:=0.056), (b:=0.14), (dx:=0.31), (dy:=0.6)))

ax0.set_aspect("equal")
ax0.set_xticks([-1., 0, 1.], minor=False)
ax0.set_xticks([-1.5, -0.5, 0.5, 1.5], minor=True)
ax0.set_yticks([-1., 0, 1.], minor=False)
ax0.set_yticks([-1.5, -0.5, 0.5, 1.5], minor=True)
ax0.tick_params(
    axis='x', which="both", direction="in", labelsize=tlfs,
    color=tcolor, length=3,
    top=True, labeltop=False,
    bottom=True, labelbottom=True,
    )
ax0.tick_params(
    axis='y', which="both", direction="in", labelsize=tlfs,
    color=tcolor, length=3,
    left=True, labelleft=True,
    right=True, labelright=False
    )
ax0.tick_params(axis="both", which="minor", length=2.)
ax0.imshow(
    bob[0].transpose(), extent=ext[0], origin="lower", cmap=cmap,
    vmin=vmin, vmax=vmax, alpha=1)
ax0.contour(
    XX[0], XX[1], bob[0].transpose(), levels=[0., df],
    colors="k", linewidths=0.5, linestyles=["dashed", "dotted"])

ax0.set_xlabel(r"$x$ ($\si{\um}$)", fontsize=lfs, labelpad=2)
ax0.set_ylabel(r"$y$ ($\si{\um}$)", fontsize=lfs, labelpad=-1)


########## Colorbar ##########
x0, x1 = ax0.get_position().x0, ax0.get_position().x1
y1 = ax0.get_position().y1
ax2 = fig.add_axes(rect=(x0, y1+ 0.05, x1-x0, 0.05))
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2,
                  orientation="horizontal",
                  ticklocation="top", extend="min", extendfrac=0.00)
ax2.tick_params(axis="x", which="both", direction="out",
                color="k", labelsize=tlfs, pad=3, rotation=0)
ax2.set_xticks([-250, -150, -50, 50, 150], minor=True)

ax2.plot([0, 0], [0., 1.], ls="dashed", lw=0.5, color="k")
ax2.plot([df, df], [0., 1.], ls="dotted", lw=0.5, color="k")

ax2.set_xlabel(
    r"$\Delta f$ ($\mathrm{kHz}/\mathrm{W}$)",
    ha="center", va="center", rotation=0, labelpad=8)


# =============================================================================
# XZ YZ CUTS
# =============================================================================

gs1 = gridspec.GridSpec(
    nrows=2, ncols=1,
    width_ratios=[1.],
    height_ratios=[1., 1.], hspace=0.0,
    figure=fig,
    left=l+dx+0.1, bottom=0.11, right=(r:=0.98), top=0.97)

ax1 = gs1.subplots(sharex=False, sharey=True, squeeze=False)

### zx, zy
for i in range(2):
    ax1[i, 0].set_aspect("equal")
    ax1[i, 0].set_xlim(*extent[2])
    ax1[i, 0].set_ylim(*extent[1])
    # tick stuff
    ax1[i, 0].set_xticks([-4, -2, 0, 2, 4], minor=False)
    ax1[i, 0].set_xticks([-3, -1, 1, 3], minor=True)
    ax1[i, 0].set_yticks([-1., 0, 1.], minor=False)
    ax1[i, 0].set_yticks([-1.5, -0.5, 0.5, 1.5], minor=True)
    ax1[i, 0].tick_params(
        axis='x', which="both", direction="in", labelsize=tlfs,
        color=tcolor, length=3,
        top=True, labeltop=False,
        bottom=True, labelbottom=False,
        )
    ax1[i, 0].tick_params(
        axis='y', which="major", direction="in", labelsize=tlfs,
        color=tcolor, length=3,
        left=True, labelleft=True,
        right=True, labelright=False,
        )
    ax1[i, 0].tick_params(
        axis="y", which="minor", direction="in", labelsize=tlfs,
        color=tcolor, length=2,
        left=True, labelleft=False,
        right=True, labelright=False
        )
    ax1[i, 0].tick_params(axis="both", which="minor", length=2)
    # display stuff
    ax1[i, 0].imshow(
        bob[i+1], extent=ext[i+1], origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax,)
    ax1[i, 0].contour(
        XX[2], XX[i], bob[i+1], levels=[0., df],
        colors="k", linewidths=0.5, linestyles=["dashed", "dotted"])

ax1[1, 0].tick_params(
    axis='x', which="both", direction="in", labelsize=tlfs,
    color=tcolor,
    top=True, labeltop=False,
    bottom=True, labelbottom=True,
    )
ax1[0, 0].set_ylabel(r"$x$ ($\si{\um}$)", fontsize=lfs, labelpad=2,
                     va="center", ha="center", rotation=90)
ax1[0, 0].yaxis.set_label_position("left")
ax1[1, 0].set_xlabel(r"$z$ ($\si{\um}$)", fontsize=lfs, labelpad=2,
                     va="top", ha="center", rotation=0)
ax1[1, 0].set_ylabel(r"$y$ ($\si{\um}$)", fontsize=lfs, labelpad=2,
                     va="bottom", ha="center", rotation=90)
ax1[1, 0].yaxis.set_label_position("left")

ax1[0, 0].spines.bottom.set_linewidth(0.4)
ax1[1, 0].spines.top.set_linewidth(0.4)


################################# SAVE FIGURE #################################
fig.savefig("./potential_difference.png")
