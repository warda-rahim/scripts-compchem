#!/usr/bin/env python3
import matplotlib.pyplot as plt;
from pylab import genfromtxt;
from matplotlib import rcParams;
from matplotlib import rc;
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from cycler import cycler

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

rcParams['figure.figsize'] = 6, 5     #adjusts the figure size
#rcParams['axes.titlepad'] = 20
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

fig, ax = plt.subplots()

seebeck = np.loadtxt("seebeck_n_type.dat")

plt.tick_params(axis='both', which='major', direction='in', length=8, labelsize=14)
plt.tick_params(axis='both', which='minor', direction='in', length=4)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200)) #position of tick marks
ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.set_xticks(np.linspace(200,1000,5)) #position of tick mark labels
ax.set_yticks(np.linspace(0,-1000,6))

default_cycler = (cycler(color=['#4b0082', '#800080', '#8a2be2', '#7b68ee', '#ee82ee']) + cycler(marker=['x', 'o', '^', 'D', 's']))
ax.set_prop_cycle(default_cycler)
plt.plot(seebeck[1:,0], seebeck[1:,1:], linewidth=1.5, markersize=8)  #plot the first column of second row on x-axis and second column onwards for all the rows from second one onwards (inclusive) on y-axis

plt.legend(seebeck[0,1:], loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.15), frameon=False, title='Carrier Concentration ($\mathregular{cm^{-3}}$)')

plt.xlabel('Temperature (K)', fontsize=14)
plt.ylabel('S ($\mathregular{{\mu}}$V $\mathregular{K^{-1}}$)', fontsize=14)

plt.subplots_adjust(left=0.15,right=0.9,bottom=0.1,top=0.9)

plt.savefig('seebeck-n-type.pdf')
plt.savefig('seebeck-n-type.png', dpi=500)
plt.show()
