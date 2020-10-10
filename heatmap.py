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

rcParams['figure.figsize'] = 6.5, 5     #adjusts the figure size
rcParams['axes.titlepad'] = 20
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

#grid = GridSpec(2,4)

fig, ax = plt.subplots()

ZT = np.loadtxt("zt_n_type.dat")

plt.tick_params(axis='both', which='major', direction='in', length=8, labelsize=14)
plt.tick_params(axis='both', which='minor', direction='in', length=4)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50)) #position of tick marks
ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
#ax.set_xticks(np.linspace(250,950,8)) #position of tick mark labels

ax.set_xlim(100,380)

plt.yscale('log')

norm = mpl.colors.Normalize(vmin=np.min(ZT[1:,1:]),vmax=np.max(ZT[1:,1:])) #normalises the colors between the max and min values of ZT 
heatmap = plt.pcolormesh(ZT[1:,0], ZT[0,1:], np.transpose(ZT[1:,1:]), shading='gouraud', antialiased='true', norm=norm, cmap='RdPu') 
cbar = plt.colorbar(heatmap, norm=norm)  #normalises the colorbar between the max and min values of ZT

max = np.max(ZT[1:,1:])   #obtains the max value from an array
maxround = '{:.3f}'.format(max)
print(max)
print(np.where(ZT[1:,1:]==max))
T = ZT[14,0]
Tround = '{:.0f}'.format(T)
C = ZT[0,59]
cround = np.format_float_scientific(C,precision=2,exp_digits=2).replace('e+','x10^') 


cround = cround.split('x10^')

ax.text(0.5, 0.25, r'Max ZT = $\bf{{{}}}$ ({} K, ${}x10^{{{}}}\ cm^{{-3}}$)'.format(maxround,Tround,cround[0],cround[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, fontweight='bold')  #print max ZT with its T & conc on the plot

plt.xlabel('Temperature (K)', fontsize=16)
plt.ylabel('Carrier concentrations ($\mathregular{cm^{-3}}$)', fontsize=16)
plt.title('Figure of Merit ZT as a Function of Temperature and Carrier Concentration', fontsize=12, fontweight='bold')

plt.subplots_adjust(left=0.15,right=0.9,bottom=0.1,top=0.9)

plt.savefig('zt-n-type.pdf')
plt.savefig('zt-n-type.png', dpi=500)
plt.show()
