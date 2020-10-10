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

rcParams['figure.figsize'] = 5, 5     #adjusts the figure size
mpl.rcParams['axes.linewidth'] = 1.5  #adjusts matplotlib border frame (i.e. axes width)

import yaml
with open("band.yaml", 'r') as f:
    data = yaml.load(f)     #data consists of file band1.yaml

file_qpoints = [] #empty array
freq = []

file_qpoints.append([i['q-position'] for i in data['phonon']])  #creates an array consisting of an array with all the qpts 
for point in data['phonon']:
    freq.append([e['frequency'] for e in point['band']])
 #creates an array consisting of arrays of frequencies at each q-point

d = 0.02 #d is a small number
qpoints = [[], [], [], [], [], [], [], []] #array of 8 points close to gamma-point (assuming a cube to account for directionality
qpoints.append([[0.5 * d, 0.5 * d, 0.5 * d], [0.5 * d, -0.5 * d, 0.5 * d], [0.5 * d, 0.5 * d, -0.5 * d], [0.5 * d, -0.5 * d, -0.5 * d], [-0.5 * d, -0.5 * d, -0.5 * d], [-0.5 * d, 0.5 * d, -0.5 * d], [-0.5 * d, -0.5 * d, 0.5 * d], [-0.5 * d, 0.5 * d, 0.5 * d]])

a = 0.2238805985 * 2 * np.pi   #to get the reciprocal lattice vector
b = 0.2238805985 * 2 * np.pi
c = 0.3042509642 * 2 * np.pi

#to get lengths of vectors from gamma(0,0,0) to each one of the eight points
for j in qpoints:
    x = []
    x.append([np.sqrt((e[0] * a) ** 2 + (e[1] * b) ** 2 + (e[2] * c) ** 2) for e in j])
print(x)

def get_symmetry_equivalent_qpoints(symmops, qpoint, tol=1e-2):
    """
    Creates a list of symmetry equivalent qpoints.
    Args:
        symmops: symmetry operations (e.g. from Pymatgen)
        qpoint:  qpoint
        tol:     tolerance
    """
    points = np.dot(qpoint, [m.rotation_matrix for m in symmops])
    rm_list = []
    # identify and remove duplicates from the list of equivalent q-points:
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            if np.allclose(pbc_diff(points[i], points[j]), [0, 0, 0], tol):
                rm_list.append(i)
                break
    return np.delete(points, rm_list, axis=0)


import numpy as np
import pymatgen.symmetry.analyzer as pmg
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

struct = Poscar.from_file('POSCAR').structure      
sg = SpacegroupAnalyzer(struct)
symmops = sg.get_point_group_operations(cartesian=False)

new_qpoints = []
for q in qpoints:
  equiv_qpoints = get_symmetry_equivalent_qpoints(symmops, q) 
  all_dists = [np.min(np.sum(np.power(k - equiv_qpoints, 2), axis=1)) for k in file_qpoints] 
  min_id = all_dists.index(np.min(all_dists))
  new_qpoints.append(file_qpoints[min_id])

print(new_qpoints)

freq_near_G = []
for i in new_qpoints:
    for j in range in (0, 3):
        freq_near_G.append(freq[min_id][j])

print(freq_near_G)
