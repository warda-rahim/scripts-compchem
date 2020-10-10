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
import h5py
import re
from natsort import natsorted

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

rcParams['figure.figsize'] = 6, 5    #adjusts the figure size
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

fig, ax = plt.subplots() 

temperatures, kappa_tensors = None, None

my_list = []
from os import walk
for path, direc, file in walk("."):
    for name in file:
        if name.endswith('.hdf5'):
            my_list.append(name)
my_list = natsorted(my_list)
print(my_list)

data = []
for n in my_list:
    data.append(h5py.File("{}".format(n), 'r')) 


temperatures = data[0]['temperature'][:]

kappa_tensors = []
for i in data:
    kappa_tensors.append(i['kappa'][:])

plt.tick_params(axis='both', which='both', direction='in', length=8, labelsize=16)
plt.xlim(0,1000)
#plt.ylim(0,1.2)
plt.xlabel('Temperature (K)', fontsize=18)
plt.ylabel(r'$\kappa$ ($W m^{-1} K^{-1}$)', fontsize=18)

kmeshes = ["5x5x7", "7x7x9", "9x9x11", "11x11x14", "13x13x15", "15x15x20", "17x17x23", "19x19x26"]
directions = ["x","y","z"]
label = []
for i in kmeshes:
    for j in directions:
        label.append("{}-{}".format(i,j))

colours = plt.cm.viridis(np.linspace(0,1,8))
markers = ["o", "x", "D"]

for x in range(len(data)):    
    for y in range(3):
        plt.plot(temperatures[5:], kappa_tensors[x][5:,y], linewidth=1.5, markevery=4, color=colours[x], marker=markers[y])

plt.legend(label, frameon=False, fontsize=10)

index = np.where(temperatures==300)
klatt_at_rt = []
for i in range(0,len(my_list)):
    klatt_at_rt.append(kappa_tensors[i][index[0][0],:3])
print("klatt_at_rt:", klatt_at_rt)

klatt_at_rt_avg = []
for i in klatt_at_rt:
    klatt_at_rt_avg.append(i[0:3].mean())
print("klatt_at_rt_avg:", klatt_at_rt)

plt.savefig('kappa_convergence_xyz.pdf')
plt.savefig('kappa_convergence_xyz.png', dpi=500)
plt.show()
