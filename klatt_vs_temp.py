#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(
         description='Plots lattice thermal conductivity against temperature.\
                      Requires a Phono3py kappamxyz.hdf5.')
parser.add_argument('-k', '--kappafile', metavar='kappamxyz.hdf5', \
                     help='path to Phono3py kappamxyz.hdf5 file')
parser.add_argument('-xm', '--xmax', metavar='maximum xaxis value', type=int, \
                    default=1000, \
                    help='maximum xaxis (temperature) value in K.')
parser.add_argument('--kcolours', metavar='kappa colour', nargs='+',
                    help='colours for the kappa-vs-temp plot lines')
parser.add_argument('--xmarkers', metavar='marker', nargs='+', type=float, 
                    help='marks a given point on the line \
                          corresponding to certain x values.')
parser.add_argument('--ymarkers', metavar='marker', nargs='+',type=float, 
                    help='marks a given point on the line \
                          corresponding to certain y values.')
parser.add_argument('-o', '--output', metavar="output file suffix", 
                    help='suffix for output filename e.g. \
                    structure name and qpoint mesh used')
parser.add_argument('-z', action='store_true',
                    help='dark mode')

args = parser.parse_args()



import matplotlib.pyplot as plt
from pylab import genfromtxt
from matplotlib import rcParams, rc
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from cycler import cycler
import h5py
import os
from scipy.interpolate import interp1d


fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')


mpl.rcParams['axes.linewidth'] = 2  
rcParams['figure.figsize'] = 12.6, 12    

def plot_kappa(axis, hdf5file, xmax, colours, xmarkers):
    """Plots lattice thermal conductivity from a Phono3py hdf5 file.
    Args:
        hdf5File (:obj:'dict'): Phono3py kappa-mxxx.hdf5 data.
        xmax (:obj: 'float'): Maximum xaxis value.
        colours (:obj: 'str'): Colours for the plot lines.
    
    Returns:
           :obj:'matplotlib.pyplot': Axis with DoS.
    """

    data = h5py.File(hdf5file, 'r')
    temperatures = data['temperature'][:]
    kappa_tensors = data['kappa'][:]


    default_cycler = (cycler(color=colours) + cycler(marker=['^', 'o', 'D', 'x']) + cycler(markersize=[20, 16, 16, 16]))
    axis.set_prop_cycle(default_cycler)
    axis.plot(temperatures[4:np.where(temperatures==xmax)[0][0]+1], kappa_tensors[4:np.where(temperatures==xmax)[0][0]+1,:3], linewidth=4, markevery=2)
    axis.plot(temperatures[4:np.where(temperatures==xmax)[0][0]+1], np.mean(kappa_tensors[4:np.where(temperatures==xmax)[0][0]+1,:3], axis=1), 
    linewidth=4, markevery=2)
    axis.legend(['$\mathregular{\kappa_{xx}}$','$\mathregular{\kappa_{yy}}$','$\mathregular{\kappa_{zz}}$','$\mathregular{\kappa_{iso}}$'], frameon=False, fontsize=48)


    axis.tick_params(axis='both', which='major', direction='in', length=24, labelsize=50, pad=15)
    axis.tick_params(axis='both', which='minor', direction='in', length=12, pad=15)
    #axis.set_xticks(range(0,xmax+1,100))
    #axis.set_yticks(np.linspace(0,1.2,5))
    axis.xaxis.set_major_locator(ticker.MaxNLocator(4))
    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axis.yaxis.set_major_locator(ticker.MaxNLocator(4))
    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    xlim = axis.set_xlim(50,xmax)
    ymax = np.max(np.mean(kappa_tensors[4:np.where(temperatures==xmax)[0][0]+1,:3], axis=1))
    ymin = np.min(np.mean(kappa_tensors[4:np.where(temperatures==xmax)[0][0]+1,:3], axis=1))
    ylim = axis.set_ylim(ymin * 0.5 , ymax * 0.9)
    axis.set_xlabel('Temperature (K)', fontsize=50)
    axis.set_ylabel(r'$\mathregular{\kappa_{l}(\lambda) \ (W \ m^{-1} \ K^{-1}}$)', fontsize=50)


    index_rt = np.where(temperatures == 300)
    print("kappa_at_rt:", kappa_tensors[index_rt[0],:3])
    print("kappa_avg_at_rt:", np.mean(kappa_tensors[index_rt[0],:3]))


    f1 = []
    for i in range(3):
        f1.append(interp1d(temperatures[:], kappa_tensors[:,i]))
    kappa_xmarker = []
    print("temp", "kappa")
    for j in xmarkers:
        x = []
        for i in f1:
             print(j, i(j))
             x.append(i(j))
        kappa_xmarker.append(x)
    
    print("temp","kappa_av")
    kappa_av_xmarker = []
    for i, j in zip(xmarkers, kappa_xmarker):
        print(i, np.mean(j))
        kappa_av_xmarker.append(np.mean(kappa_xmarker))
    for x, y in zip(xmarkers, kappa_av_xmarker):
        axis.plot([x, x, xlim[0]], [ylim[0], y, y], c='black', linestyle=':', linewidth=2.5, marker="None")


    return axis


fig, ax = plt.subplots(figsize=(12.6,12))

ax = plot_kappa(ax, args.kappafile, args.xmax, args.kcolours, args.xmarkers)

plt.subplots_adjust(left=0.18, right=0.94, bottom=0.15, top=0.95)

plt.savefig('klatt_vs_temp-{}.pdf'.format(args.output))
plt.savefig('klatt_vs_temp-{}.png'.format(args.output), dpi=500)
plt.show()
