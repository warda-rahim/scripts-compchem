from pymatgen import Structure
from pymatgen.core.surface import SlabGenerator

structure = Structure.from_file('POSCAR_conv')

structure.add_oxidation_state_by_element({"Ca":2, "Bi":-3, "O":-2})
#structure.add_oxidation_state_by_guess()

# These are distances in Angstroms
dist = [10,15,20,25,30]
# We iterate through the distances twice, once for vac, once for slab
for vac in dist:
    for thickness in dist:
        slabgen = SlabGenerator(structure, miller_index=(0,0,1), 
                                min_slab_size=thickness, min_vacuum_size=vac, lll_reduce=True)
        slabs = slabgen.get_slabs()

        dipole_free_slabs = []
        for slab in slabs:
            if not slab.is_polar():
                dipole_free_slabs.append(slab)
        
        slab = dipole_free_slabs[0] # <-- put a number in here! 
        slab.to(fmt='poscar', filename='slab_{}_{}.vasp'.format(thickness,vac))
