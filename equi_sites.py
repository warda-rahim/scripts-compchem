#!/usr/bin/env python3
from pymatgen.io.vasp.inputs import Poscar
from bsym.interface.pymatgen import unique_structure_substitutions
from pymatgen import Lattice, Structure
import numpy as np
from bsym import ConfigurationSpace
from bsym import Configuration
from bsym.interface.pymatgen import unique_structure_substitutions
from ase import io, Atoms
import bsym.interface.pymatgen as bpy
import os
import ase
structure = Poscar.from_file("POSCAR").structure
supercell = structure * [1, 2, 1]
unique_structures = unique_structure_substitutions( supercell, 'O', { 'S':1, 'O':111 } )
len( unique_structures )
for i in range(len(unique_structures)):
    unique_structures[i].remove_species(["S"])
    unique_structures[i].sort(key=None, reverse=False)
    unique_structures[i].to(filename="POSCAR_{}".format(i),fmt="poscar")
