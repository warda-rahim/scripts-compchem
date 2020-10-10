#!/bin/bash

for slab in *.vasp; do 
    id=${slab%.vasp}
    mkdir $id
    mv $slab $id/POSCAR
    cp INCAR POTCAR KPOINTS job $id
done 
