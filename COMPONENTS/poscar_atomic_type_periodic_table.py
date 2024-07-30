from includes import *


def atom_to_coords(name):
    atom_coords = {
        "H": [0, 0, 0],
        "He": [0, 0, 1],
        "Li": [1, 0, 0],
        "Be": [1, 0, 1],
        "B": [1, 6, 0],
        "C": [1, 6, 1],
        "N": [1, 7, 0],
        "O": [1, 7, 1],
        "F": [1, 8, 0],
        "Ne": [1, 8, 1],
        "Na": [2, 0, 0],
        "Mg": [2, 0, 1],
        "Al": [2, 6, 0],
        "Si": [2, 6, 1],
        "P": [2, 7, 0],
        "S": [2, 7, 1],
        "Cl": [2, 8, 0],
        "Ar": [2, 8, 1],
        "K": [3, 0, 0],
        "Ca": [3, 0, 1],
        "Sc": [3, 1, 0],
        "Ti": [3, 1, 1],
        "V": [3, 2, 0],
        "Cr": [3, 2, 1],
        "Mn": [3, 3, 0],
        "Fe": [3, 3, 1],
        "Co": [3, 4, 0],
        "Ni": [3, 4, 1],
        "Cu": [3, 5, 0],
        "Zn": [3, 5, 1],
        "Ga": [3, 6, 0],
        "Ge": [3, 6, 1],
        "As": [3, 7, 0],
        "Se": [3, 7, 1],
        "Br": [3, 8, 0],
        "Kr": [3, 8, 1],
        "Rb": [4, 0, 0],
        "Sr": [4, 0, 1],
        "Y": [4, 1, 0],
        "Zr": [4, 1, 1],
        "Nb": [4, 2, 0],
        "Mo": [4, 2, 1],
        "Tc": [4, 3, 0],
        "Ru": [4, 3, 1],
        "Rh": [4, 4, 0],
        "Pd": [4, 4, 1],
        "Ag": [4, 5, 0],
        "Cd": [4, 5, 1],
        "In": [4, 6, 0],
        "Sn": [4, 6, 1],
        "Sb": [4, 7, 0],
        "Te": [4, 7, 1],
        "I": [4, 8, 0],
        "Xe": [4, 8, 1],
        "Cs": [5, 0, 0],
        "Ba": [5, 0, 1],
        "Lu": [5, 1, 0],
        "Hf": [5, 1, 1],
        "Ta": [5, 2, 0],
        "W": [5, 2, 1],
        "Re": [5, 3, 0],
        "Os": [5, 3, 1],
        "Ir": [5, 4, 0],
        "Pt": [5, 4, 1],
        "Au": [5, 5, 0],
        "Hg": [5, 5, 1],
        "Tl": [5, 6, 0],
        "Pb": [5, 6, 1],
        "Bi": [5, 7, 0],
        "Po": [5, 7, 1],
        "At": [5, 8, 0],
        "Rn": [5, 8, 1],
        "Fr": [6, 0, 0],
        "Ra": [6, 0, 1],
        "Lr": [6, 1, 0],
        "Rf": [6, 1, 1],
        "Db": [6, 2, 0],
        "Sg": [6, 2, 1],
        "Bh": [6, 3, 0],
        "Hs": [6, 3, 1],
        "Mt": [6, 4, 0],
        "Ds": [6, 4, 1],
        "Rg": [6, 5, 0],
        "Cn": [6, 5, 1],
        "Nh": [6, 6, 0],
        "Fl": [6, 6, 1],
        "Mc": [6, 7, 0],
        "Lv": [6, 7, 1],
        "Ts": [6, 8, 0],
        "Og": [6, 8, 1],
        "La": [5, 9, 0],
        "Ce": [5, 9, 1],
        "Pr": [5, 10, 0],
        "Nd": [5, 10, 1],
        "Pm": [5, 11, 0],
        "Sm": [5, 11, 1],
        "Eu": [5, 12, 0],
        "Gd": [5, 12, 1],
        "Tb": [5, 13, 0],
        "Dy": [5, 13, 1],
        "Ho": [5, 14, 0],
        "Er": [5, 14, 1],
        "Tm": [5, 15, 0],
        "Yb": [5, 15, 1],
        "Ac": [6, 9, 0],
        "Th": [6, 9, 1],
        "Pa": [6, 10, 0],
        "U": [6, 10, 1],
        "Np": [6, 11, 0],
        "Pu": [6, 11, 1],
        "Am": [6, 12, 0],
        "Cm": [6, 12, 1],
        "Bk": [6, 13, 0],
        "Cf": [6, 13, 1],
        "Es": [6, 14, 0],
        "Fm": [6, 14, 1],
        "Md": [6, 15, 0],
        "No": [6, 15, 1],
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
