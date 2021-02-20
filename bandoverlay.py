#!/usr/bin/env python3
import argparse
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

parser = argparse.ArgumentParser(
         description='Overlays electronic band structures with and without\
                      spin-orbit-coupling (SOC) for comparion \
                      (Note: The valence band maxima must not be set to 0 eV \
                             in .dat files for a comparison')
parser.add_argument('-b', '--band', metavar='band.dat file',
                    help='band.dat file from an electronic band structure \
                          calculation performed without SOC')
parser.add_argument('--bandsoc', metavar='bandsoc.dat file (SOC)',
                     help='band.dat file from an electronic band structure \
                        calculation performed with SOC')
parser.add_argument('-d', '--distances', metavar='kpoint distances', nargs='+', \
                    type=float,
                    help='distances at which kpoints/xaxis-labels are located')
parser.add_argument('-l', '--labels', metavar='xaxis labels', nargs='+',
                    help='labels for the xaxis (kpoints)')
parser.add_argument('-c', '--colours', metavar='band colours', nargs='+',
                    default=['#FF1493', '#06B800'],
                    help='colours for each band structure')
parser.add_argument('--ymin', metavar='bottom yaxis limit',
                    type=float,
                    help='bottom limit for the y-axis')
parser.add_argument('--ymax', metavar='top yaxis limit',
                    type=float,
                    help='top limit for the y-axis')
parser.add_argument('-o', '--output', metavar='output file',
                    help='suffix for the output file name')
args = parser.parse_args()



mpl.rcParams['axes.linewidth'] = 2  #adjusts matplotlib border frame (i.e. axes width)
rcParams['figure.figsize'] = 12.6, 12     #adjusts the figure size

fig, ax = plt.subplots()


band = np.loadtxt(args.band)
bandsoc = np.loadtxt(args.bandsoc)


# we need to plot individual band lines instead of all the points getting plotted (as qpts are repeated for each band in band.dat file (sumo output). 
# (Note: In the .dat files used for this script, there is an empty line between each set of kpoints)

kpt = 0
with open(args.band) as filename:
    for line in filename:
        if line.strip():
            kpt += 1
        else:
            break

a =[]
for i in range(0, len(band), kpt-1):       
    a.append(band[i:(i+kpt-1),1])

b = []
for i in range(0, len(bandsoc), kpt-1):
    b.append(bandsoc[i:(i+kpt-1),1])



colours = args.colours

ax.plot(bandsoc[0:kpt-1,0], np.transpose(b), linewidth=2.5, color=colours[0])
ax.plot(band[0:kpt-1,0], np.transpose(a), linewidth=2.5,  linestyle='--', color=colours[1])



distances = args.distances
labels = args.labels
labels = [r'$\Gamma$' if x=='G' else x for x in labels]

plt.xticks(distances, labels)                
for i in distances:
    plt.axvline(x=i, color='#000000', linewidth=2.5)


ax.tick_params(axis='both', which='both', direction='in', length=24, labelsize=50, pad=15)

if args.ymin:
    ax.set_ylim(bottom=args.ymin)
if args.ymax:
    ax.set_ylim(top=args.ymax)
ax.set_xlim(0, band[:,0][-1])      

ax.set_ylabel('Energy (eV)', fontsize=50)

plt.subplots_adjust(left=0.15, right=0.97, bottom=0.07, top=0.96)

fig.savefig('band-overlay-{}.pdf'.format(args.output) if args.output is not None else 'band-overlay.pdf')
fig.savefig('band-overlay-{}.png'.format(args.output) if args.output is not None else 'band-overlay.png', dpi=500)
#plt.show()
