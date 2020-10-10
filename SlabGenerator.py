from pymatgen.core.surface import generate_all_slabs
from pymatgen import Structure 
structure = Structure.from_file('POSCAR_conv')

#structure.add_oxidation_state_by_guess()
structure.add_oxidation_state_by_element({"Ca":2, "Bi":-3, "O":-2})

all_slabs = generate_all_slabs(structure, max_index=1, min_slab_size=15, min_vacuum_size=15, lll_reduce=True)

for slab in all_slabs:
    print(slab.miller_index,slab.dipole)

dipole_free_slabs = []
for slab in all_slabs:
    if not slab.is_polar():
        dipole_free_slabs.append(slab)

for n,slab in enumerate(dipole_free_slabs):
    slab.to(fmt='poscar', filename='slab_number_{}.vasp'.format(n))
