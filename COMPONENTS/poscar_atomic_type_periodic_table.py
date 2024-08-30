from includes import *


def atom_to_coords(name):
    atom_coords = {
        "H": [0, 0, 1],
        "He": [0, 0, 0],
        "Li": [1, 0, 1],
        "Be": [1, 3, 1],
        "B": [1, 3, 0],
        "C": [1, 2, 1],
        "N": [1, 2, 0],
        "O": [1, 1, 1],
        "F": [1, 1, 0],
        "Ne": [1, 0, 0],
        "Na": [2, 0, 1],
        "Mg": [2, 3, 1],
        "Al": [2, 3, 0],
        "Si": [2, 2, 1],
        "P": [2, 2, 0],
        "S": [2, 1, 1],
        "Cl": [2, 1, 0],
        "Ar": [2, 0, 0],
        "K": [3, 0, 1],
        "Ca": [3, 3, 1],
        "Sc": [3, 8, 1],
        "Ti": [3, 8, 0],
        "V": [3, 7, 1],
        "Cr": [3, 7, 0],
        "Mn": [3, 6, 1],
        "Fe": [3, 6, 0],
        "Co": [3, 5, 1],
        "Ni": [3, 5, 0],
        "Cu": [3, 4, 1],
        "Zn": [3, 4, 0],
        "Ga": [3, 3, 0],
        "Ge": [3, 2, 1],
        "As": [3, 2, 0],
        "Se": [3, 1, 1],
        "Br": [3, 1, 0],
        "Kr": [3, 0, 0],
        "Rb": [4, 0, 1],
        "Sr": [4, 3, 1],
        "Y": [4, 8, 1],
        "Zr": [4, 8, 0],
        "Nb": [4, 7, 1],
        "Mo": [4, 7, 0],
        "Tc": [4, 6, 1],
        "Ru": [4, 6, 0],
        "Rh": [4, 5, 1],
        "Pd": [4, 5, 0],
        "Ag": [4, 4, 1],
        "Cd": [4, 4, 0],
        "In": [4, 3, 0],
        "Sn": [4, 2, 1],
        "Sb": [4, 2, 0],
        "Te": [4, 1, 1],
        "I": [4, 1, 0],
        "Xe": [4, 0, 0],
        "Cs": [5, 0, 1],
        "Ba": [5, 3, 1],
        "La": [5, 15, 1],
        "Ce": [5, 15, 0],
        "Pr": [5, 14, 1],
        "Nd": [5, 14, 0],
        "Pm": [5, 13, 1],
        "Sm": [5, 13, 0],
        "Eu": [5, 12, 1],
        "Gd": [5, 12, 0],
        "Tb": [5, 11, 1],
        "Dy": [5, 11, 0],
        "Ho": [5, 10, 1],
        "Er": [5, 10, 0],
        "Tm": [5, 9, 1],
        "Yb": [5, 9, 0],
        "Lu": [5, 8, 1],
        "Hf": [5, 8, 0],
        "Ta": [5, 7, 1],
        "W": [5, 7, 0],
        "Re": [5, 6, 1],
        "Os": [5, 6, 0],
        "Ir": [5, 5, 1],
        "Pt": [5, 5, 0],
        "Au": [5, 4, 1],
        "Hg": [5, 4, 0],
        "Tl": [5, 3, 0],
        "Pb": [5, 2, 1],
        "Bi": [5, 2, 0],
        "Po": [5, 1, 1],
        "At": [5, 1, 0],
        "Rn": [5, 0, 0],
        "Fr": [6, 0, 1],
        "Ra": [6, 3, 1],
        "Ac": [6, 15, 1],
        "Th": [6, 15, 0],
        "Pa": [6, 14, 1],
        "U": [6, 14, 0],
        "Np": [6, 13, 1],
        "Pu": [6, 13, 0],
        "Am": [6, 12, 1],
        "Cm": [6, 12, 0],
        "Bk": [6, 11, 1],
        "Cf": [6, 11, 0],
        "Es": [6, 10, 1],
        "Fm": [6, 10, 0],
        "Md": [6, 9, 1],
        "No": [6, 9, 0],
        "Lr": [6, 8, 1],
        "Rf": [6, 8, 0],
        "Db": [6, 7, 1],
        "Sg": [6, 7, 0],
        "Bh": [6, 6, 1],
        "Hs": [6, 6, 0],
        "Mt": [6, 5, 1],
        "Ds": [6, 5, 0],
        "Rg": [6, 4, 1],
        "Cn": [6, 4, 0],
        "Nh": [6, 3, 0],
        "Fl": [6, 2, 1],
        "Mc": [6, 2, 0],
        "Lv": [6, 1, 1],
        "Ts": [6, 1, 0],
        "Og": [6, 0, 0],
    }

    return atom_coords[name]


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

        atom_rep = atom_to_coords(atoms[atom_type])
        rw = [0.0] * 7
        cl = [0.0] * 16
        sp = [0.0] * 1
        rw[atom_rep[0]] = 1.0
        cl[atom_rep[1]] = 1.0
        sp[0] = atom_rep[2]  # the spin is a single bit, so we drop it in one thing
        # we have the whole representation right now - after this i add it into
        # the graph for clarity - the ends of the poscar may have extra data
        nodes_array.append([*cl, *rw, *sp])
    return nodes_array
