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
from matplotlib.colors import ListedColormap
from cycler import cycler

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')


parser = argparse.ArgumentParser(
         description='Plots phonon group velocities vs frequency')
parser.add_argument('-k', '--kappa', metavar='kappa file path',
                    help='path to Phono3py kappa-mxyz.hdf5 file')
#parser.add_argument('-t', '--temp', metavar='temperature', type=int,
#                     default=300,
#                    help='temperature to find lifetimes and mode kappa at a specific temperature value')
#parser.add_argument('-m', '--mesh', metavar='qpoint mesh', nargs='+', type=int,
#                     help='qpoint mesh required for some versions of Phono3py')
parser.add_argument('--colour', metavar='waterfall colour', nargs='+', default='',
                    help='colours for the waterfall plots')
parser.add_argument('--cmin', metavar='colour', default='#E75480',
                    help='first colour for the colourmap')
parser.add_argument('--cmax', metavar='colour', default='#A6D608',
                    help='second colour for the colourmap')
parser.add_argument('--alpha', metavar='opacity', type=float,
                    default=1,
                    help='opacity for line colours')
parser.add_argument('--density', metavar='number of colours', type=int,
                    default=512,
                    help='number of colours to be created between cmax and cmin')
parser.add_argument('-o', '--output', metavar='output file suffix', default='',
                     help='suffix for the output filename')
parser.add_argument('--style', metavar='style sheet', nargs='+', default='',
                    help='style sheets to use. Later ones will \
                          override earlier ones if they conflict.')
parser.add_argument('-z', action='store_true',
                    help='dark mode')
args = parser.parse_args()


fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

mpl.rcParams['axes.linewidth'] = 2 #adjusts matplotlib border frame (i.e. axes width)
rcParams['figure.figsize'] = 12.6, 12   #adjusts the figure size

fig, ax = plt.subplots() 

temperatures, kappa_tensors = None, None

f = h5py.File(args.kappa, 'r') 
qpoints = f['qpoint'][:]
temperatures = f['temperature'][:]
frequency = f['frequency'][:]
group_velocity = f['group_velocity'][:]
nbands = frequency.shape[1]

print(frequency.shape)
print(group_velocity.shape)

freq_1d = frequency.flatten()

print(group_velocity)

#tot_grp_vel = []
#for i in group_velocity:
#    b  = []
#    for j in i:
#        a = np.sqrt(j[0]**2 + j[1]**2 + j[2]**2)
#        b.append(a)
#    tot_grp_vel.append(b)
#


tot_grp_vel = np.array(np.sqrt(np.square(group_velocity[:,:,0]) +
            np.square(group_velocity[:,:,1]) + np.square(group_velocity[:,:,2])))

#tot_grp_vel = np.array(tot_grp_vel)
print(tot_grp_vel.shape)

group_velocity_1D = tot_grp_vel[:, :].flatten()
group_velocity_1D = group_velocity_1D * 100  #converts group_vel from THz.Angstrom to metres per second
group_velocity_1D = np.abs(group_velocity_1D)

print("min-grp-vel:", np.min(group_velocity_1D))
print("max-grp-vel:", np.max(group_velocity_1D))

ax.tick_params(axis='both', which='major', direction='in', length=24, width=2, labelsize=50)
ax.tick_params(axis='both', which='minor', direction='in', length=12, width=2)
#ax.set_xticks(np.linspace(0,1000,6))
#ax.set_yticks(np.linspace(0,1.2,5)) 
ax.set_yscale('log')
ax.set_ylim(1e0,1e4)
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.set_xlabel('Frequency (THz)', fontsize=50)
ax.set_ylabel(r'$\mathregular{|\nu_{\lambda}| (m s^{-1})}$', fontsize=50)


def generate_cmap(cmin, cmax, alpha, density):
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


#viridis = cm.viridis(np.linspace(0, 1, nbands))
#print(viridis)

colors = generate_cmap(args.cmin, args.cmax, args.alpha, args.density)
colormap = colors(np.linspace(0, 1, nbands))
newcolors = np.tile(colormap, [len(qpoints),1])

plt.scatter(freq_1d, group_velocity_1D, s=5, facecolor=newcolors) 

plt.subplots_adjust(left = 0.15, right = 0.97, top  = 0.95, bottom = 0.15)
plt.savefig('group_vel_vs_freq-{}.pdf'.format(args.output))
plt.savefig('group_vel_vs_freq-{}.png'.format(args.output), dpi=500)
#plt.show()
