#!/bin/bash

for mposcar in MPOSCAR-*; do
   id=${mposcar#MPOSCAR-}
   mkdir $id
   mv $mposcar $id/POSCAR
   cp INCAR KPOINTS POTCAR job $id
done      
