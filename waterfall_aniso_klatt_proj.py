#! /usr/bin/env python3

import argparse
import h5py
import numpy as np
import math

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
         description='Plots anisotropic modal contributions to group velocities \
                      and mean free paths at a specific temperature \
                      with mode kappa projected onto the data. \
                      Also, prints the maximum and average mean free path along each direction. \
                      (NOTE: You can change the script to set ylim and tick positions on xaxis.)')
parser.add_argument('-k', '--kappa', metavar='kappa file path',
                    help='path to Phono3py kappa-mxyz.hdf5 file')
parser.add_argument('-d', '--directions', metavar='directions', nargs='+',
                     default=['x', 'y', 'z'],
                     help='directions for anisotropic data. Accepts x-z or a-c') 
parser.add_argument('-t', '--temp', metavar='temperature', type=int,
                     default=300,
                    help='temperature to find lifetimes and mode kappa at a specific temperature value')
parser.add_argument('-m', '--mesh', metavar='qpoint mesh', nargs='+', type=int,
                     default='',
                     help='qpoint mesh required for some versions of Phono3py')
parser.add_argument('--percent', metavar='percentage', type=float, default=1,
                   help='percentage of data cut from the bottom \
                         for colour bar normalisation')
parser.add_argument('--bottom', metavar='percentage', type=float, default=1,
                    help='percentage of data cut from the bottom to set yaxis limits \
                          for each quantity plotted on the y-axis')
parser.add_argument('--top', metavar='percentage', type=float, default=0.1,
                    help='percentage of data cut from the top to set yaxis limits \
                          for each quantity plotted on the y-axis')
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
parser.add_argument('--colourbar', action='store_true', 
                    help='plot colourbar next to each subplot')
parser.add_argument('-o', '--output', metavar='output file suffix', default='',
                     help='suffix for the output filename')
args = parser.parse_args()



f = h5py.File(args.kappa, 'r')
qpoints = f['qpoint'][:]
temperatures = f['temperature'][:]
frequency = f['frequency'][:]
nbands = frequency.shape[1]
mode_kappa = f['mode_kappa'][:]

freq_1d = frequency.flatten()

temp_index = np.where(temperatures == args.temp)
mode_kappa_1D = []
for i in range(3):
    mode_kappa_2D = mode_kappa[temp_index[0][0], :, :, i]
    mode_kappa_1D.append(mode_kappa_2D.flatten() / (args.mesh[0] * args.mesh[1] * args.mesh[2]))



dir_dict = {'x': 0, 'y': 1, 'z': 2, 
            'a': 0, 'b': 1, 'c': 2}

quantities = ['group velocity', 'mean free path']
directions = args.directions
for d in directions:
    assert d in dir_dict, '{} must be x-z or a-c'.format(d)



data = []

group_velocity = f['group_velocity'][:]
group_velocity_1D = []
for i in range(3):
    grp_vel = group_velocity[:,:,i]
    grp_vel_1D = grp_vel.flatten()
    grp_vel_1D = grp_vel_1D * 100 # converts group_vel from THz.Angstrom to metres per second
    group_velocity_1D.append(np.abs(grp_vel_1D))
data.append(group_velocity_1D)


gamma = f['gamma'][:]
gamma_1D = gamma[temp_index[0][0], :, :].flatten()
lifetimes = [1.0 / (2 * 2 * np.pi * i) for i in gamma_1D] # gives lifetimes in ps


lifetime_sec = np.array(lifetimes) * 1e-12  # converts lifetimes from ps to s
mfp = []
for i in range(3):
    mfp.append([j*k for j, k in zip(lifetime_sec, group_velocity_1D[i])])
data.append(mfp)



# prints maximum mfp along each direction
mfp_no_nan = []
for i in range(len(mfp)):
    mfp_no_nan.append([j for j in mfp[i] if not math.isnan(j)])

for i in range(len(directions)):
    print('max mfp along {}:'.format(directions[i]), '{:.3g}'.format(np.max(mfp_no_nan[i])))
print('avg mfp:', '{:.3g}'.format(np.mean(mfp_no_nan)))




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


# we are going to use the same normalisation object for each subplot
mode_kappa_avg = np.mean(mode_kappa[temp_index[0][0], :, :, :3], axis=2) / (args.mesh[0] * args.mesh[1] * args.mesh[2]) 
mode_kappa_avg_1D_sort = sorted(mode_kappa_avg.flatten())
mode_kappa_99_percent = mode_kappa_avg_1D_sort[int(len(mode_kappa_avg_1D_sort) * args.percent/100) : len(mode_kappa_avg_1D_sort)]
cnorm = colors.LogNorm(vmin=min(mode_kappa_99_percent), vmax=max(mode_kappa_99_percent))



mpl.rcParams['axes.linewidth'] = 2.5

# Defining figsize for different scenario, for example for plotting the data for either all 3 or 2 or 1 direction
figsize_dict = {'1' : (12.6, 24),
                '2' : (27, 24),
                '3' : (38, 24),
                '1_cb' : (15, 24),
                '2_cb': (32, 24),
                '3_cb': (44, 24)}

if args.colourbar:
    key = str(len(directions)) + '_cb'
else:
    key = str(len(directions))
fig = plt.figure(figsize=figsize_dict[key])


grid = gs.GridSpec(2, len(args.directions))
ax = [' '] * (2 * len(args.directions))

label_dict = {'group velocity' : r'$\mathregular{|\nu_{\lambda}|\ (m s^{-1})}$',
              'mean free path': r'$\mathregular{\Lambda_{\lambda}\ (m)}$'}

idx = 0
for i in range(len(quantities)):
    for j in range(len(directions)):
        ax[idx] = fig.add_subplot(grid[idx])
        sca = ax[idx].scatter(freq_1d, data[i][dir_dict[directions[j]]], c=mode_kappa_1D[j], cmap=colours, norm=cnorm, s=15)
        
        # Setting the ylim
        data2 = sorted(data[i][j]) 
        new_data = data2[int(len(data2) * args.bottom/100) : int(len(data2) * (1 - args.top/100))]
        ax[idx].set_ylim(min(new_data), max(new_data))
        
        ax[idx].set_ylabel(label_dict[quantities[i]], fontsize=50)
        if args.colourbar:
            cbar = plt.colorbar(sca, extend='min')
            cbar.ax.tick_params(which='major', direction='in', length=12, width=3, labelsize=32)
            cbar.ax.tick_params(which='minor', direction='in', length=8, width=3)
            cbar.set_label(r'$\mathregular{\kappa_{\lambda} \ (W \ m^{-1} \ K^{-1}}$)', fontsize=32)
        idx += 1



# Manually setting the ylim for each plot instead of going for the percentage (i.e. cutting data from top and bottom)
#ax[0].set_ylim(1e1, 1e4)
#ax[1].set_ylim(1e1, 1e4)
#ax[2].set_ylim(1e1, 1e4)
#ax[3].set_ylim(1e-12, 1e-6)
#ax[4].set_ylim(1e-12, 1e-6)
#ax[5].set_ylim(1e-12, 1e-6)


for i in range(len(ax)):
    ax[i].tick_params(axis='both', which='major', pad=15, direction='in', length=24, width=2.5, labelsize=50)
    ax[i].tick_params(axis='both', which='minor', pad=15, direction='in', length=12, width=2.5)
    ax[i].set_yscale('log')
    ax[i].set_xlabel('Frequency (THz)', fontsize=50)
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax[i].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

if len(directions) == 1:
    grid.update(left=0.18, wspace=0.4, right=0.96, bottom=0.08, top=0.97)
elif len(directions) == 2:
    grid.update(left=0.12, wspace=0.4, right=0.96, bottom=0.08, top=0.97)
elif len(directions) == 3:
    grid.update(left=0.08, wspace=0.3, right=0.98, bottom=0.09, top=0.97)

plt.savefig('waterfall-aniso-{}K-{}.pdf'.format(args.temp, args.output))
plt.savefig('waterfall-aniso-{}K-{}.png'.format(args.temp, args.output))

