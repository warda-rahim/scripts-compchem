#!/usr/bin/env python3
import matplotlib.pyplot as plt;
from pylab import genfromtxt;
from matplotlib import rcParams;
from matplotlib import rc;
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

rcParams['axes.titlepad'] = 20        #adjusts space between title and plot
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)
rcParams['figure.figsize'] = 12.6,12     #adjusts the figure size

fig, ax = plt.subplots()

band = np.loadtxt("band-a2-hse.dat")
bandsoc = np.loadtxt("band-a2-hse+soc.dat")

ax.tick_params(axis='both', which='both', direction='in', length=8, labelsize=14)

# plot individual bands for each set of qpoints
a =[]
for i in range(0, len(band), 223):       
    a.append(band[i:(i+223),1])

b = []
for i in range(0, len(bandsoc), 223):
    b.append(bandsoc[i:(i+223),1])

ax.plot(bandsoc[0:223,0], np.transpose(b), linewidth=1.5, color='#FFA500')
ax.plot(band[0:223,0], np.transpose(a), linewidth=1.5,  linestyle='--', color='#FF1493')


distances = [0.0000, 0.8312, 1.3185, 1.8057, 2.0599, 2.8911, 3.3783, 3.6325]
labels = [r'$\Gamma$', 'Y', 'V', r'$\Gamma$', 'A', 'M', 'L', 'V']
plt.xticks(distances, labels)
for i in distances:
    plt.axvline(x=i, color='#000000', linewidth=1.5)


ax.tick_params(axis='both', which='major', direction='in', length=24, labelsize=50, pad=15)
ax.tick_params(axis='both', which='minor', direction='in', length=12, pad=15)
ax.set_ylim(-6,8)
ax.set_xlim(0, band[:,0][-1])      


ax.set_ylabel('Energy (eV)', fontsize=50)

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)

fig.savefig('band-overlay.pdf')
fig.savefig('band-overlay.png', dpi=500)
plt.show()
