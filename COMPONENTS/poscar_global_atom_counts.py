from includes import *


def atomToCoords(name):
    atom_coords = {
        "H": [0, 0, 0], "He": [0, 0, 1],
        "Li": [1, 0, 0], "Be": [1, 0, 1], "B": [1, 6, 0], "C": [1, 6, 1],
        "N": [1, 7, 0], "O": [1, 7, 1], "F": [1, 8, 0], "Ne": [1, 8, 1],
        "Na": [2, 0, 0], "Mg": [2, 0, 1], "Al": [2, 6, 0], "Si": [2, 6, 1],
        "P": [2, 7, 0], "S": [2, 7, 1], "Cl": [2, 8, 0], "Ar": [2, 8, 1],
        "K": [3, 0, 0], "Ca": [3, 0, 1], "Sc": [3, 1, 0], "Ti": [3, 1, 1],
        "V": [3, 2, 0], "Cr": [3, 2, 1], "Mn": [3, 3, 0], "Fe": [3, 3, 1],
        "Co": [3, 4, 0], "Ni": [3, 4, 1], "Cu": [3, 5, 0], "Zn": [3, 5, 1],
        "Ga": [3, 6, 0], "Ge": [3, 6, 1], "As": [3, 7, 0], "Se": [3, 7, 1],
        "Br": [3, 8, 0], "Kr": [3, 8, 1], "Rb": [4, 0, 0], "Sr": [4, 0, 1],
        "Y": [4, 1, 0], "Zr": [4, 1, 1], "Nb": [4, 2, 0], "Mo": [4, 2, 1],
        "Tc": [4, 3, 0], "Ru": [4, 3, 1], "Rh": [4, 4, 0], "Pd": [4, 4, 1],
        "Ag": [4, 5, 0], "Cd": [4, 5, 1], "In": [4, 6, 0], "Sn": [4, 6, 1],
        "Sb": [4, 7, 0], "Te": [4, 7, 1], "I": [4, 8, 0], "Xe": [4, 8, 1],
        "Cs": [5, 0, 0], "Ba": [5, 0, 1], "Lu": [5, 1, 0], "Hf": [5, 1, 1],
        "Ta": [5, 2, 0], "W": [5, 2, 1], "Re": [5, 3, 0], "Os": [5, 3, 1],
        "Ir": [5, 4, 0], "Pt": [5, 4, 1], "Au": [5, 5, 0], "Hg": [5, 5, 1],
        "Tl": [5, 6, 0], "Pb": [5, 6, 1], "Bi": [5, 7, 0], "Po": [5, 7, 1],
        "At": [5, 8, 0], "Rn": [5, 8, 1], "Fr": [6, 0, 0], "Ra": [6, 0, 1],
        "Lr": [6, 1, 0], "Rf": [6, 1, 1], "Db": [6, 2, 0], "Sg": [6, 2, 1],
        "Bh": [6, 3, 0], "Hs": [6, 3, 1], "Mt": [6, 4, 0], "Ds": [6, 4, 1],
        "Rg": [6, 5, 0], "Cn": [6, 5, 1], "Nh": [6, 6, 0], "Fl": [6, 6, 1],
        "Mc": [6, 7, 0], "Lv": [6, 7, 1], "Ts": [6, 8, 0], "Og": [6, 8, 1],
        "La": [5, 9, 0], "Ce": [5, 9, 1], "Pr": [5, 10, 0], "Nd": [5, 10, 1],
        "Pm": [5, 11, 0], "Sm": [5, 11, 1], "Eu": [5, 12, 0], "Gd": [5, 12, 1],
        "Tb": [5, 13, 0], "Dy": [5, 13, 1], "Ho": [5, 14, 0], "Er": [5, 14, 1],
        "Tm": [5, 15, 0], "Yb": [5, 15, 1], "Ac": [6, 9, 0], "Th": [6, 9, 1],
        "Pa": [6, 10, 0], "U": [6, 10, 1], "Np": [6, 11, 0], "Pu": [6, 11, 1],
        "Am": [6, 12, 0], "Cm": [6, 12, 1], "Bk": [6, 13, 0], "Cf": [6, 13, 1],
        "Es": [6, 14, 0], "Fm": [6, 14, 1], "Md": [6, 15, 0], "No": [6, 15, 1]
    }
    return atom_coords[name]

max_atom_counts = {
    "Ta": 84,
    "Ga": 92,
    "P": 132,
    "S": 150,
    "O": 240,
    "Li": 96,
    "Cu": 96,
    "Ca": 72,
    "Mn": 64,
    "Co": 104,
    "Cr": 56,
    "Sn": 104,
    "Ba": 84,
    "Y": 56,
    "Mg": 149,
    "Fe": 100,
    "Pr": 50,
    "Si": 232,
    "Ti": 100,
    "H": 292,
    "Na": 96,
    "Pu": 62,
    "Nb": 64,
    "Ce": 46,
    "Sc": 116,
    "N": 144,
    "La": 46,
    "In": 96,
    "Ni": 90,
    "Bi": 96,
    "Cl": 128,
    "K": 81,
    "Te": 112,
    "I": 112,
    "V": 68,
    "Rh": 68,
    "Sb": 72,
    "Zr": 64,
    "Ho": 42,
    "Ge": 172,
    "Er": 90,
    "C": 256,
    "F": 192,
    "Rb": 64,
    "Ag": 80,
    "Se": 112,
    "Al": 119,
    "Pd": 120,
    "B": 144,
    "Xe": 36,
    "Sr": 72,
    "Mo": 72,
    "W": 50,
    "As": 112,
    "Pt": 48,
    "Ru": 48,
    "Dy": 42,
    "Ir": 76,
    "Hf": 54,
    "Yb": 72,
    "Zn": 216,
    "Lu": 56,
    "Eu": 32,
    "Au": 82,
    "Pb": 76,
    "Be": 102,
    "U": 36,
    "Sm": 52,
    "Hg": 116,
    "Cs": 72,
    "Tl": 108,
    "Nd": 52,
    "Th": 40,
    "Re": 50,
    "Cd": 141,
    "Tb": 48,
    "Tc": 36,
    "Gd": 40,
    "Os": 28,
    "Tm": 30,
    "Br": 120,
    "Pm": 32,
    "Np": 24,
    "Pa": 12,
    "Ac": 16,
    "Am": 16,
    "Po": 6,
    "Kr": 20,
    "Ra": 1,
    "Ar": 2,
    "He": 8,
    "Ne": 14,
    "Rn": 1,
}

def encode_count(count, size):
    if size < count:
        raise Exception(f"Count of {count} doesn't fit in size {size}")
    return [1]*count + [0]*(size-count)

max_atoms_in_row = [292, 256, 232, 216, 141, 116, 62]

max_atoms_in_column = [292, 116, 84, 100, 120, 216, 256, 240, 192, 46, 52, 62, 40, 48, 90, 72]

def extract_element_counts(poscar):
    m_elements = poscar[5].split()
    m_counts = list(map(int, poscar[6].split()))
    return dict(zip(m_elements, m_counts))

def info(poscar):
    atoms = extract_element_counts(poscar=poscar)
    atoms_in_row    = [0] * len(max_atoms_in_row)
    atoms_in_column = [0] * len(max_atoms_in_column)

    # Put atoms in rows and columns
    for a in atoms: # for each atom type
        element = a
        count   = atoms[a]
        row, column, spin = atomToCoords(element)
        atoms_in_row[row] += count
        atoms_in_column[column] += count

    # Assemble the input array
    encoded_atoms = []
    for row, rowmax in zip(atoms_in_row, max_atoms_in_row):
        encoded_atoms.append(encode_count(row, rowmax))
    for col, colmax in zip(atoms_in_column, max_atoms_in_column):
        encoded_atoms.append(encode_count(col, colmax))

    return flatten(encoded_atoms)


