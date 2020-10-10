#! /usr/bin/env python3

import argparse
## Arguments
 
parser = argparse.ArgumentParser(
         description='Uses pymatgen to find crystallographically\
                      unique atoms sites in a structure')
parser.add_argument('-s', '--struc', nargs='+',
                     help='name of structure file')

args = parser.parse_args()

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

structures = []
for x in args.struc:
    structures.append(Poscar.from_file(x).structure)

nsites = []
for y in structures:
    ana = SpacegroupAnalyzer(y)
    struct= ana.get_symmetrized_structure()
    nsites.append(len(struct.equivalent_sites))

with open('unique_sites.txt', 'w') as f:
    for i, j in zip(args.struc, nsites):
        print(i, ":", j, file=f)
