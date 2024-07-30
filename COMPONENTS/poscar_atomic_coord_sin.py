from includes import *


def info(poscar):
    poscar[5].split()
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

        nodes_array.append(
            flatten([get_relative_coordinates(i) for i in unpack_line(poscar[8 + i])])
        )

    return nodes_array
