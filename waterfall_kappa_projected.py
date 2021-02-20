#! /usr/bin/env python3

import argparse
import h5py
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import genfromtxt
from matplotlib import rcParams, rc, cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gs
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from cycler import cycler

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')


parser = argparse.ArgumentParser(
         description='Plots modal contributions to group velocities, lifetimes, \
                      and mean free paths at a specific temperature \
                      with mode-kappa projected on the data')
parser.add_argument('-k', '--kappa', metavar='kappa file',
                    help='Phono3py kappa-mxyz.hdf5 file')
parser.add_argument('-t', '--temp', metavar='temperature', type=int,
                     default=300,
                    help='temperature at which to find lifetimes and mode kappas')
parser.add_argument('-m', '--mesh', metavar='qpoint mesh', nargs='+', type=int,
                     default='',
                     help='qpoint mesh required for some versions of Phono3py')
parser.add_argument('--percent', metavar='percentage', type=float, default=1,
                   help='percentage of data not included for colour bar normalisation')
parser.add_argument('--colour', metavar='waterfall colour', nargs='+', default='',
                    help='colours for the waterfall plots')
parser.add_argument('--cmin', metavar='colour', default='#A6D608',
                    help='first colour for the colourmap')
parser.add_argument('--cmax', metavar='colour', default='#E75480',
                    help='second colour for the colourmap')
parser.add_argument('--alpha', metavar='opacity', type=float,
                    default=1,
                    help='opacity for line colours')
parser.add_argument('--density', metavar='number of colours', type=int,
                    default=512,
                    help='number of colours to be created between cmax and cmin')
parser.add_argument('-o', '--output', metavar='output file suffix', default='',
                     help='suffix for the output filename')
args = parser.parse_args()


file = h5py.File(args.kappa, 'r')
qpoints = file['qpoint'][:]
temperatures = file['temperature'][:]
frequency = file['frequency'][:]
mode_kappa = file['mode_kappa'][:]
group_velocity = file['group_velocity'][:]
gamma = file['gamma'][:]
mode_kappa = file['mode_kappa'][:]
nbands = frequency.shape[1]


freq_1d = frequency.flatten()


temp_index = np.where(temperatures == args.temp)
mode_kappa_2D = mode_kappa[temp_index[0][0], :, :, :3]
mode_kappa_2D_avg = np.mean(mode_kappa_2D, axis=2) / (args.mesh[0] * args.mesh[1] * args.mesh[2]) 
mode_kappa_1D = mode_kappa_2D_avg.flatten()


tot_grp_vel = np.array(np.sqrt(np.square(group_velocity[:,:,0]) +
            np.square(group_velocity[:,:,1]) + np.square(group_velocity[:,:,2])))

group_velocity_1D = tot_grp_vel[:, :].flatten()
group_velocity_1D = group_velocity_1D * 100  #converts group_vel from THz.Angstrom to metres per second and takes the absolute value
group_velocity_1D = np.abs(group_velocity_1D)


gamma_1D = gamma[temp_index[0][0], :, :].flatten()
lifetimes = [1.0 / (2 * 2 * np.pi * i) for i in gamma_1D]


lifetime_sec = np.array(lifetimes) * 1e-12
mfp = []
for i, j in zip(lifetime_sec, group_velocity_1D):
    mfp.append(i*j)



def generate_cmap(cmin='#A6D608', cmax='#E75480', alpha=1, density=512):
    """Generates single-gradient colour maps.

    Args:
        cmax (:obj:'str'): colour at maximum.
        cmin (:obj:'str'): colour at minimum.
        density (:obj:'int'): number of colours to output.

    Returns:
        (:obj:'ListedColormap'): The colourmap.
    """

    colours = np.full((density, 4), float(alpha))
    for n in range(0,3):
        cmin2 = int(cmin[2*n+1:2*n+3], 16)
        cmax2 = int(cmax[2*n+1:2*n+3], 16)
        colours[:,n] = np.linspace(cmin2/256, cmax2/256, density)
    cmap = ListedColormap(colours)

    return cmap


colours = generate_cmap(args.cmin, args.cmax, args.alpha, args.density)


mode_kappa_sorted = sorted(mode_kappa_1D)
mode_kappa_new = mode_kappa_sorted[int(len(mode_kappa_sorted) * args.percent/100) : len(mode_kappa_sorted)]
cnorm = colors.LogNorm(vmin=min(mode_kappa_new), vmax=max(mode_kappa_new))

mpl.rcParams['axes.linewidth'] = 2.5
fig = plt.figure(figsize=(12.6, 36)) # Edit at your peril.
grid = gs.GridSpec(3, 1)
ax = [' ', ' ', ' ']

#ax[0] = fig.add_subplot(grid[0])
#ax[0].scatter(freq_1d, mode_kappa_1D, c=mode_kappa_1D, cmap=colours, norm=cnorm, s=15)
#ax[0].set_ylim(1e-9,1e-1)
#ax[0].set_ylabel(r'$\mathregular{\kappa_{\lambda} \ (W \ m^{-1} \ K^{-1}}$)', fontsize=50, labelpad=15)

ax[0] = fig.add_subplot(grid[0])
ax[0].scatter(freq_1d, group_velocity_1D, c=mode_kappa_1D, cmap=colours, norm=cnorm, s=15)
ax[0].set_ylim(1e1,1e4)
ax[0].set_ylabel(r'$\mathregular{|\nu_{\lambda}| (m s^{-1})}$', fontsize=50)

ax[1] = fig.add_subplot(grid[1])
ax[1].scatter(freq_1d, lifetimes, c=mode_kappa_1D, cmap=colours, norm=cnorm, s=15)
ax[1].set_ylim(1e-2,1e2)
ax[1].set_ylabel(r'$\mathregular{\tau_{\lambda} (ps)}$', fontsize=50)

ax[2] = fig.add_subplot(grid[2])
ax[2].scatter(freq_1d, mfp, c=mode_kappa_1D, cmap=colours, norm=cnorm, s=15)
ax[2].set_ylim(1e-12,1e-6)
ax[2].set_ylabel(r'$\mathregular{\Lambda_{\lambda} (m)}$', fontsize=50, labelpad=15)

for i in range(3):
    ax[i].tick_params(axis='both', which='major', pad=15, direction='in', length=24, width=2.5, labelsize=50)
    ax[i].tick_params(axis='both', which='minor', pad=15, direction='in', length=12, width=2.5)
    ax[i].set_yscale('log')
    ax[i].set_xlabel('Frequency (THz)', fontsize=50)
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax[i].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))


grid.update(left=0.21, right=0.97, bottom=0.05, top=0.95, hspace=0.25) # Likewise.
plt.savefig('waterfall-{}K-{}.pdf'.format(args.temp, args.output))
plt.savefig('waterfall-{}K-{}.png'.format(args.temp, args.output))

