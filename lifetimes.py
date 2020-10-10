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
         description='Plots phonon lifetimes vs frequency at a specific temperature')
parser.add_argument('-k', '--kappa', metavar='kappa file path',
                    help='path to Phono3py kappa-mxyz.hdf5 file')
parser.add_argument('-t', '--temp', metavar='temperature', type=int,
                     default=300,
                    help='temperature to find lifetimes and mode kappa at a specific temperature value')
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


mpl.rcParams['axes.linewidth'] = 2 #adjusts matplotlib border frame (i.e. axes width)
rcParams['figure.figsize'] = 12.6, 12   #adjusts the figure size

fig, ax = plt.subplots() 

file = h5py.File(args.kappa, 'r') 
qpoints = file['qpoint'][:]
temperatures = file['temperature'][:]
frequency = file['frequency'][:]
gamma = file['gamma'][:]
nbands = frequency.shape[1]
print(frequency.shape)
print(gamma.shape)

freq_1D = frequency.flatten()

index = np.where(temperatures == args.temp)
gamma_1D = gamma[index[0][0], :, :].flatten()
lifetimes = [1.0 / (2 * 2 * np.pi * i) for i in gamma_1D]

print("min-lifetime:", np.min(lifetimes))
print("max-lifetime:", np.max(lifetimes))



ax.tick_params(axis='both', which='both', direction='in', length=24, width=2, labelsize=50)
ax.tick_params(axis='both', which='minor', direction='in', length=12, width=2)
#ax.set_xticks(np.linspace(0,1000,6))
#ax.set_yticks(np.linspace(0,1.2,5)) 
ax.set_yscale('log')
ax.set_ylim(1e-1,2e1)
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.set_xlabel('Frequency (THz)', fontsize=50)
ax.set_ylabel(r'$\mathregular{\tau_{\lambda} (ps)}$', fontsize=50)



def generate_cmap(cmin='#E75480', cmax='#A6D608', alpha=1, density=512):
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

plt.scatter(freq_1D, lifetimes, s=5, facecolor=newcolors) 

plt.subplots_adjust(left = 0.16, right = 0.97, top  = 0.95, bottom = 0.15)
plt.savefig('lifetimes_vs_freq-{}K-{}.pdf'.format(args.temp, args.output))
plt.savefig('lifetimes_vs_freq-{}K-{}.png'.format(args.temp, args.output), dpi=500)
plt.show()
