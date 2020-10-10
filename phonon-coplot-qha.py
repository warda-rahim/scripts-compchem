#!/usr/bin/env python3
import matplotlib.pyplot as plt;
from pylab import genfromtxt;
from matplotlib import rcParams;
from matplotlib import rc;
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import re
from natsort import natsorted

mpl.rcParams.update(mpl.rcParamsDefault)
rcParams['figure.figsize'] = 5, 5     #adjusts the figure size
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

my_list = []
from os import walk
for path, direc, file in walk("."):
    for name in file:
        if name.endswith('.yaml'):
            my_list.append(name)
print(my_list)

my_new_list = natsorted(my_list, signed=True)

data = []  
import yaml
for i in my_new_list:
    with open("{}".format(i), 'r') as f:
        data.append(yaml.safe_load(f))

dists = []
eigenvalues = []   

for j in range(11):
    dists.append([i['distance'] for i in data[j]['phonon']])
print(len(dists))

for j in range(11):
    x = []
    for point in data[j]['phonon']: 
        x.append([e['frequency'] for e in point['band']])
    eigenvalues.append(x)
print(len(eigenvalues))

l = [] 
for i in data[0]['phonon']: 
    if 'label' in i:
        l.append(i['label'])
l = [r'$\Gamma$' if x=='G' else x for x in l]
print(l)


p = []
for j in range(11):
    p.append([i['distance'] for i in data[j]['phonon'] if 'label' in i])
print(len(p))
print(p[0])

y = 0
for n in range(len(p)-1):
    for x, value in enumerate(dists[0]): 
        while not ((value <= p[n+1][y+1]) and (value >= p[n+1][y])) and y < (len(p)-1):    
            y += 1   
        dists[n+1][x] = (p[n+1][y+1] - p[n+1][y])/(p[n][y+1] - p[n][y]) * dists[n][x]

plt.axhline(linewidth=1.5, color='black', linestyle='--')

for x in p[0]:
    plt.axvline(x=x, linewidth=1.5, color='black')

plt.xticks(p[0], l)               
plt.tick_params(axis='both', which='both', direction='in', length=8, labelsize=16)
plt.xlim(0, dists[0][-1])
plt.ylabel('Frequency (THz)', fontsize=18)

c = iter(plt.cm.viridis(np.linspace(0,1,11)))

for i, j in zip(dists, eigenvalues):
    lines = plt.plot(i, j, linewidth=0.5, color=next(c))

#plt.legend(iter(lines), ('+5%', '+4%', '+3%', '+2%', '+1%', '0%', '-1%', '-2%', '-4%', '-5%'))

#plt.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.9)

plt.tight_layout()
plt.savefig('phonon-volume.pdf')
plt.savefig('phonon-volume.png', dpi=500)
plt.show()
