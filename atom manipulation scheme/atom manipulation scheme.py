# -*- coding: utf-8 -*-
"""
A scheme for atomic state manipulation in an experimental setup.
"""

from itertools import combinations_with_replacement

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import colors
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = \
    r"\usepackage{siunitx}" \
    r"\usepackage{physics}" \
    r"\usepackage{bm}"
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

DPI = 500 # dots per inch

mm = 1/25.4 # To set the size of the figure in mm rather than inch

########## Colors in plots ##########
def tinting(color: np.ndarray, tint: float)-> np.ndarray:
    """
    Tint a color (make it lighter). 0 <= tint <= 1
    tint = 1 -> white, tint = 0 -> color unchanged
    """
    return tint + (1-tint) * color

ms = 2. # marker size
mew = 0.5 # marker edge width
elw = 0.6 # error linewidth
## colors
ncolors = np.empty((6, 4, 4), dtype=float)
# raw colors
ncolors[0, 0, :] = colors.to_rgba("blueviolet")
ncolors[1, 0, :] = colors.to_rgba("tab:red")
ncolors[2, 0, :] = colors.to_rgba("tab:green")
ncolors[3, 0, :] = colors.to_rgba("coral")
ncolors[4, 0, :] = colors.to_rgba("blue")
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

arrowProps = {
    'arrowstyle': "->,head_width=0.15,head_length=0.3",
    'connectionstyle': "arc3,rad=0.",
    'facecolor': "k",
    'edgecolor': "k",
    'shrinkA': 0.8,
    'shrinkB': 0.,
    'lw': 0.8}

arrowProps1 = {
    'arrowstyle': "<->,head_width=0.12,head_length=0.24",
    'connectionstyle': "arc3,rad=0.",
    'facecolor': "k",
    'edgecolor': "k",
    'shrinkA': 0.0,
    'shrinkB': 0.,
    'lw': 0.6}


def curly_arrow(xy_start: tuple,
                xy_end: tuple,
                n: int = 5,
                w_arrow: float = 0.1,
                w_wave: float = 0.1,
                **pathkw):
    """
    Curly arrow patch
    Axes aspect must be "equal" for a nice arrow: Axes.set_aspect("equal")

    Parameters
    ----------
    xy_start, xy_end : tuple
        Extremities of the arrow (start: arrow side)
    n : int, optional
        Number of oscillations. The default is 5.
    w_arrow : float, optional
        Width of the arrow. The default is 0.1.
    w_wave : float, optional
        Width of the wave. The default is 0.1.
    **pathkw : TYPE
        matplotlib Path kwargs (colors, etc).

    Returns
    -------
    patch : matplotlib patch
        Axes.add_patch(arrow) to plot it on the Axes.

    """
    x1, y1 = xy_start
    x2, y2 = xy_end
    L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    n0 = L / (2 * np.pi)
    
    phi = np.pi/20
    dx = L/10
    e = 0.6
    k = 201
    
    N = 4 + (4 + k)*2
    x = np.empty(N, dtype=float)
    y = np.empty(N, dtype=float)
    codes = np.empty(N, dtype=object)
    
    # the arrow
    x[0:4] = [0, dx, dx, dx]
    y[0:4] = [0, w_arrow, -w_arrow, 0]
    codes[0:4] = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    
    # straight appendice
    x[4:8] = [dx, dx*(1 + e - 0.1), dx*(1 + e), dx*(1 + e)]
    y[4:8] = [0, 0, 0, w_wave*np.sin(phi)]
    codes[4:8] = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3]
    x[N-4:] = [dx*(1 + e), dx*(1 + e), dx*(1 + e - 0.1), dx]
    y[N-4:] = [w_wave*np.sin(phi), 0, 0, 0]
    codes[N-4:] = [Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
    
    # wave
    x[8:k+8] = np.linspace(dx*(1+e), L, k)
    x[k+8:2*k+8] = x[k+7:7:-1]
    y[8:k+8] = w_wave * np.sin(n*np.linspace(0, L, k)/n0 + phi)
    y[k+8:2*k+8] = y[k+7:7:-1]
    codes[8:2*k+8] = Path.LINETO
    
    ang = np.arctan2(y2 - y1, x2 - x1)
    x_ang = np.cos(ang) * x - np.sin(ang) * y + x1
    y_ang = np.cos(ang) * y + np.sin(ang) * x + y1
    
    verts = np.array([x_ang, y_ang]).transpose()
    path = Path(verts, codes)
    patch = patches.PathPatch(path, **pathkw)
    return patch


# =============================================================================
# PLOT - 
# =============================================================================

## Initialize the figure
fig = plt.figure(num=0,
                 figsize=(140*mm, 95*mm),
                 dpi=DPI,
                 clear=True,
                 facecolor="w")

fig.subplots_adjust(left=0.01, right=0.99,
                    bottom=0.01, top=0.99)

ax = fig.add_axes(rect=(0.0, 0.0, 1., 1.), facecolor="0.9")

ax.axis("off")
ax.set_aspect("equal")
ax.set_xlim(0, 10.5)
ax.set_ylim(1.9, 9.52)


# =============================================================================
# Levels
# =============================================================================

lvl_lw =0.8
X0 = 2.6
dx = 0.3
ddx = 0.1
dy = 0.16

dx0 = 2*dx+ddx

########## 5S Levels ##########
yF2 = 2.9

ax.text(X0+1., yF2-2*dy+0.16, r"$\mathbf{5S_{1/2}}$",
          ha="center", va="center", fontsize=lfs)
ax.text(X0+1., yF2-2*dy-0.16, r"$\bm{F}\mathbf{=\!2}$",
          ha="center", va="center", fontsize=lfs)
for i in range(1, 5):
    ax.plot([X0-0.2*dx-(i*dx0-dx)*0.8, X0-0.2*dx-(i*dx0+dx)*0.8],
             [yF2-i*dy, yF2-i*dy],
              color="k", lw=lvl_lw)
    ax.text(X0-0.2*dx - (i*dx0)*0.8+0.02, yF2-i*dy-0.14, f"${2-i}$",
              ha="center", va="center", fontsize=tlfs)
ax.plot([X0-dx, X0+dx],
         [yF2, yF2],
          color="k", lw=lvl_lw)
ax.text(X0 + 0.02, yF2-0.14, "$2$",
          ha="center", va="center", fontsize=tlfs)


########## 6P ##########
yP = 4.4
D = 0.3
ax.plot([X0-dx, X0+dx], [yP, yP], color="k", lw=lvl_lw)
ax.text(X0-1.1, yP, r"$\mathbf{6P_{3/2}}$",
          ha="center", va="center", fontsize=lfs)
ax.plot([X0-dx, X0+dx], [yP+D, yP+D], 
         linestyle=(0., (2., 2.)), color="k", lw=0.6)


########## 52D ##########
y52D = 5.8
ax.plot([X0-dx, X0+dx], [y52D, y52D], color="k", lw=lvl_lw)
ax.text(X0-1.1, y52D+0.16, r"$\mathbf{52D_{5/2}}$",
          ha="center", va="center", fontsize=lfs)
ax.text(X0-1.1, y52D-0.16, r"$m_j = +\frac{5}{2}$",
          ha="center", va="center", fontsize=lfs)


########## 52F ##########
y52F = 7.
ax.plot([X0-dx, X0+dx], [y52F, y52F], color="k", lw=lvl_lw)
ax.text(X0-1.1, y52F+0.16, r"$\mathbf{52F}$",
          ha="center", va="center", fontsize=lfs)
ax.text(X0-1.1, y52F-0.16, r"$m_l = +2$",
          ha="center", va="center", fontsize=lfs)


########## stark manifold ##########
y52C = 7.5 # y position of the circular state
X1 = 8.8 # x position of the circular state
dx = 0.28
ddx = 0.15
dy = 0.19
dx0 = 2*dx+ddx

### m = 48..51
for i, j in combinations_with_replacement(range(1+(k:=3)), 2):
    ax.plot([X1+(i-k)*dx0-dx, X1+(i-k)*dx0+dx],
             [y52C + (i-2*j+k)*dy, y52C + (i-2*j+k)*dy],
             color="k", lw=0.8, zorder=3.1)

dx1 = 4.5*dx
X2 = X1 - 3*dx0 - dx1
dy1 = dy * dx1 / (2*dx + ddx)
### m = 3
for i in range(2):
    ax.plot([X2-dx, X2+dx],
             [y52C - (2*i-3)*dy + dy1, y52C - (2*i-3)*dy + dy1],
             color="k", lw=0.8)
ax.plot([X2-dx, X2+dx],
         [y52C - dy - dy1, y52C - dy - dy1],
         color="k", lw=0.8)
### m = 2
for i in range(2):
    ax.plot([X2-dx0-dx, X2-dx0+dx],
             [y52C - (2*i-4)*dy + dy1, y52C - (2*i-4)*dy + dy1],
             color="k", lw=0.8)
### m = 1
for i in range(3):
    ax.plot([X2-2*dx0-dx, X2-2*dx0+dx],
             [y52C - (2*i-5)*dy + dy1, y52C - (2*i-5)*dy + dy1],
             color="k", lw=0.8)
### m = 0
for i in range(3):
    ax.plot([X2-3*dx0-dx, X2-3*dx0+dx],
             [y52C - (2*i-6)*dy + dy1, y52C - (2*i-6)*dy + dy1],
             color="k", lw=0.8)

### dotted lines
lstyle = (0, (2, 3))
# m = 2
ax.plot([X2-dx0, X2-dx0],
         [y52C + dy1, y52C - dy1 - dy - 0.1],
         color="k", linestyle=lstyle, lw=0.6)
# m = 3
ax.plot([X2, X2],
        [y52C + 0.28, y52C - 0.28],
         color="k", linestyle=lstyle, lw=0.6)
# m = 3 -> 48
ax.plot([X2 + 0.4, X2 + dx1 - 0.4],
        [y52C + 3*dy - dy1/dx1*0.35 + dy1, y52C + 3*dy + dy1/dx1*0.35],
         color="k", linestyle=lstyle, lw=0.6)
ax.plot([X2 + 0.4, X2 + dx1 - 0.4],
        [y52C - 3*dy + dy1/dx1*0.35 - dy1, y52C - 3*dy - dy1/dx1*0.35],
         color="k", linestyle=lstyle, lw=0.6)

### line m = ...
ym = 9.4
for i in range(4):
    ax.text(X1-i*(2*dx+ddx), ym, r"$m\!=\!" f"{51-i}" r"$",
             ha="center", va="center", fontsize=7)
for i in range(4):
    ax.text(X2-i*dx0, ym, r"$m\!=\!" f"{3-i}" r"$",
             ha="center", va="center", fontsize=7)

### k = ...
dyk = 0.16
# m = 49..51
for i, j in combinations_with_replacement(range(1+(k:=2)), 2):
    if 2*j-i-2 > 0:
        txt = r"$k\!=\!\text{-}" f"{abs(int(2*j-i-2))}" r"$"
    else:
        txt = r"$k\!=\!" f"{abs(int(2*j-i-2))}" r"$"
    ax.text(X1+(i-k)*dx0, y52C + (i-2*j+k)*dy + dyk, txt,
             ha="center", va="center", fontsize=7)
# m = 48
for i in range(2):
    if i == 0:
        txt = r"$k\!=\!\text{-}3$"
    else:
        txt = r"$k\!=\!3$"
    ax.text(X1-3*dx0, y52C + 3*(2*i-1)*dy + dyk, txt,
             ha="center", va="center", fontsize=7)
# m = 2, 3
for i in range(2):
    txt = r"$k\!=\!\text{-}" f"{48-i}" r"$"
    ax.text(X1-(3+i)*dx0-dx1, y52C-dy1 - (3+i)*dy + dyk, txt,
             ha="center", va="center", fontsize=7)

########## 52C ##########
ax.plot([X1-dx, X1+dx], [y52C, y52C], color="r", lw=1., zorder=3.1)
### m = 48..50
for i in range(3):
    ax.plot([X1-(i+1)*dx0-dx, X1-(i+1)*dx0+dx],
             [y52C-(i+1)*dy, y52C-(i+1)*dy],
             color="b", lw=lvl_lw, zorder=3.1)
### m = 3
ax.plot([X2-dx, X2+dx],
         [y52C - 3*dy - dy1, y52C - 3*dy - dy1],
         color="b", lw=0.8, zorder=3.1)
### m = 2
ax.plot([X2-dx0-dx, X2-dx0+dx],
         [y52C - 4*dy - dy1, y52C - 4*dy - dy1],
         color="b", lw=0.8, zorder=3.1)

connStyle = {
    'connectionstyle': "arc,angleA=135,angleB=-45,armA=8,armB=8,rad=4"}
ax.annotate("", xy=(X1+0.16, y52C), xytext=(X1+0.7, y52C - 0.5),
             ha="center", va="center", fontsize=7, color=rawc[1],
             arrowprops=arrowProps | connStyle)
ax.text(X1 + 0.7, y52C - 0.7, "Circular level",
        ha="center", va="center", fontsize=lfs,)
ax.text(X1 + 0.7, y52C - 1., r"$n = 52$",
        ha="center", va="center", fontsize=lfs,)

# =============================================================================
# Sequence
# =============================================================================

# Optical pumping
ax.annotate("", xy=(X0, yF2-0.3), xytext=(X0-2.3, yF2 - 4*0.16 - 0.3),
             ha="center", va="center", fontsize=7, color=rawc[2],
             arrowprops=arrowProps | {'edgecolor': rawc[2]})
ax.text(X0 - 0.8, yF2-0.8, r"\textbf{(1)}",
        ha="center", va="center", fontsize=lfs,
        color="k")

# laser 420
ax.annotate("", xy=(X0, yP+D), xytext=(X0, yF2),
             ha="center", va="center", fontsize=7, color=rawc[0],
             arrowprops=arrowProps | {'edgecolor': rawc[0]})
ax.text(X0+1.1, (yP + yF2)/2+0.15, r"$\SI{420}{\nm}$",
        ha="center", va="center", fontsize=lfs,
        color=rawc[0])
ax.text(X0+1.1, (yP + yF2)/2-0.15, r"$\sigma^+$",
        ha="center", va="center", fontsize=lfs,
        color=rawc[0])
ax.annotate("", xy=(X0+dx+0.05, yP+D), xytext=(X0+dx+0.05, yP),
             ha="center", va="center",
             arrowprops=arrowProps1)
ax.text(X0 + 0.6, yP + D/2, r"$\Delta$",
        ha="center", va="center", fontsize=lfs,
        color="k")
# laser 1015
ax.annotate("", xy=(X0, y52D), xytext=(X0, yP+D),
             ha="center", va="center", fontsize=7, color=rawc[1],
             arrowprops=arrowProps | {'edgecolor': rawc[1]})
ax.text(X0+1.1, (yP + y52D)/2+0.15, r"$\SI{1015}{\nm}$",
        ha="center", va="center", fontsize=lfs,
        color=rawc[1])
ax.text(X0+1.1, (yP + y52D)/2-0.15, r"$\sigma^+$",
        ha="center", va="center", fontsize=lfs,
        color=rawc[1])
# Numbering
mkr = MarkerStyle(r"$\bigg\{$")
mkr._transform.rotate_deg(180)
ax.plot(X0 + 1.7, (yF2 + y52D)/2 + 0.1, marker=mkr, markersize=75, c="k")
ax.text(X0 + 2.05, (yF2+y52D)/2 + 0.1, r"\textbf{(2)}",
        ha="center", va="center", fontsize=lfs,
        color="k")



# DF
arrowDF = curly_arrow(
    (X0, y52F-0.04), (X0, y52D+0.04), n=4, w_arrow=0.05, w_wave=0.07,
    ec="k", fc="k", lw=0.6)
ax.add_patch(arrowDF)
ax.text(X0-1.1, (y52D+y52F)/2, r"$\SI{64.76}{\GHz}$",
          ha="center", va="center", fontsize=lfs)
ax.text(X0 + 0.6, 6.2, r"\textbf{(3)}",
        ha="center", va="center", fontsize=lfs,
        color="k")

# Stark switch
ellipse = patches.Ellipse((3.7, 7.), 1.5, 1.3,
                          facecolor="0.9", edgecolor="k",
                          linewidth=0.6)
ax.add_patch(ellipse)
connStyle = {
    'connectionstyle': "arc,angleA=0,angleB=180,armA=70,armB=80,rad=40"}
ax.annotate("", xy=(X2-1.5*dx0, y52C-dy1 - 4*dy), xytext=(X0+0.4, y52F),
             ha="center", va="center", fontsize=7, color=rawc[1],
             arrowprops=arrowProps | connStyle)
ax.text(3.7, 7.2, "Stark\nswitching",
        ha="center", va="center", fontsize=lfs,
        color="k")

# adiabatic transfer
slope = (dy - 0.04) / (dx0 - 0.08)

ax.annotate("", xy=(X2-0.04, y52C - 3*dy - dy1 - 0.02),
            xytext=(X2-dx0+0.04, y52C - 4*dy - dy1 + 0.02),
            ha="center", va="center", fontsize=7, color=rawc[3],
            arrowprops=arrowProps | {'edgecolor': rawc[3]})
ax.plot([X2+0.04, X2+0.04+0.2],
        [y52C - 3*dy - dy1 + 0.02, y52C - 3*dy - dy1 + 0.02 + slope*0.2],
        color=rawc[3], lw=0.8)
ax.annotate("", xy=(X1-3*dx0-0.04, y52C - 3*dy - 0.02),
            xytext=(X1-3*dx0-0.04-0.24, y52C - 3*dy - 0.02 - 0.24*slope),
            ha="center", va="center", fontsize=7, color=rawc[3],
            arrowprops=arrowProps | {'edgecolor': rawc[3]})
for i in range(3):
    ax.annotate("", xy=(X1-i*dx0-0.04, y52C - i*dy - 0.02),
                xytext=(X1-(i+1)*dx0+0.04, y52C - (i+1)*dy + 0.02),
                ha="center", va="center", fontsize=7, color=rawc[3],
                arrowprops=arrowProps | {'edgecolor': rawc[3]})
ax.text(7., 5.75, r"Adiabatic transfer",
        ha="center", va="center", fontsize=lfs,
        color="k")
ax.text(6.4, 5.2, "Radio-frequency\ntransition",
        ha="center", va="center", fontsize=lfs,
        color="k")
ax.text(8.1, 5.2, r"$\SI{225}{\MHz}$",
        ha="center", va="center", fontsize=lfs,
        color="k")
mkr = MarkerStyle(r"$\bigg\{$")
mkr._transform.rotate_deg(107.5)
ax.plot(6.9, 6.5, marker=mkr, markersize=125, c="k")
ax.text(6.98, 6.2, r"\textbf{(4)}",
        ha="center", va="center", fontsize=lfs,
        color="k")

################################# SAVE FIGURE #################################
fig.savefig("./atom manipulation scheme.png")
