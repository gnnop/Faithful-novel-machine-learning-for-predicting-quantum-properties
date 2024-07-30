from includes import *


def info(poscar):
    atoms = poscar[5].split()
    numbs = poscar[6].split()

    total = 0
    for i in range(len(numbs)):
        total += int(numbs[i])
        numbs[i] = total

    nodes_array = []

    cur_indx = 0
    atom_type = 0
    for i in range(total):
        cur_indx += 1
        if cur_indx > numbs[atom_type]:
            atom_type += 1

        # micah, this is for you to add the extended database stuff into
        arr = [0.0] * 118
        arr[mendeleev.element(atoms[atom_type]).atomic_number - 1] = 1.0
        nodes_array.append(arr)
