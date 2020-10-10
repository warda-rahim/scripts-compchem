#!/usr/bin/env python3
import matplotlib.pyplot as plt;
from pylab import genfromtxt;
from matplotlib import rcParams;
from matplotlib import rc;
import matplotlib.ticker as ticker
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

rcParams['figure.figsize'] = 5, 4.5     #adjusts the figure size
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

fonts = ['Whitney Book Extended']
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': fonts})
plt.rc('text', usetex=False)
plt.rc('pdf', fonttype=42)
plt.rc('mathtext', fontset='stixsans')

data = []
import yaml
for i in range(1,3):
    with open("thermal_properties{}.yaml".format(i), 'r') as f: 
        data.append(yaml.load(f))

temp = [] 
S = [[], []]

temp.append([i['temperature'] for i in data[0]['thermal_properties']])


for j in range(2):
   for e in data[j]['thermal_properties']:
       S[j].append([e['entropy']])

S[0] = np.divide(S[0], 32)
print(S[0])
S[1] = np.divide(S[1], 4)
print(S[1])


plt.tick_params(axis='both', which='both', direction='in', length=8, labelsize=16)

plt.xlabel('Temperature (K)', fontsize=18)
plt.ylabel('Entropy (J/K/mol)', fontsize=18)

c = plt.cm.viridis(np.linspace(0,1,5)) 

#plt.subplots_adjust(0.1, 0.8, 0.1, 0.9)

plt.plot(temp[0], S[0], linewidth=2.0, color='#4B0082', label=r'$\alpha_{old}$')
plt.plot(temp[0], S[1], linewidth=1.5, color='#339F34', linestyle='-.', label=r'$\alpha_{new}$')

plt.legend(loc='center right', frameon=False, prop={'size': 16})

plt.tight_layout()
plt.savefig('entropy.pdf')
plt.savefig('entropy.png', dpi=500)
plt.show()
