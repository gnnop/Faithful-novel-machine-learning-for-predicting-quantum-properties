from includes import *
# Define the elements and their positions
elements = [
    # Period 1
    ('H', 0, 0), ('He', 17, 0),
    # Period 2
    ('Li', 0, 1), ('Be', 1, 1), ('B', 12, 1), ('C', 13, 1), ('N', 14, 1), ('O', 15, 1), ('F', 16, 1), ('Ne', 17, 1),
    # Period 3
    ('Na', 0, 2), ('Mg', 1, 2), ('Al', 12, 2), ('Si', 13, 2), ('P', 14, 2), ('S', 15, 2), ('Cl', 16, 2), ('Ar', 17, 2),
    # Period 4
    ('K', 0, 3), ('Ca', 1, 3), ('Sc', 2, 3), ('Ti', 3, 3), ('V', 4, 3), ('Cr', 5, 3), ('Mn', 6, 3), ('Fe', 7, 3),
    ('Co', 8, 3), ('Ni', 9, 3), ('Cu', 10, 3), ('Zn', 11, 3), ('Ga', 12, 3), ('Ge', 13, 3), ('As', 14, 3), ('Se', 15, 3),
    ('Br', 16, 3), ('Kr', 17, 3),
    # Period 5
    ('Rb', 0, 4), ('Sr', 1, 4), ('Y', 2, 4), ('Zr', 3, 4), ('Nb', 4, 4), ('Mo', 5, 4), ('Tc', 6, 4), ('Ru', 7, 4),
    ('Rh', 8, 4), ('Pd', 9, 4), ('Ag', 10, 4), ('Cd', 11, 4), ('In', 12, 4), ('Sn', 13, 4), ('Sb', 14, 4), ('Te', 15, 4),
    ('I', 16, 4), ('Xe', 17, 4),
    # Period 6
    ('Cs', 0, 5), ('Ba', 1, 5), ('La', 2, 5), ('Ce', 3, 9), ('Pr', 4, 9), ('Nd', 5, 9), ('Pm', 6, 9), ('Sm', 7, 9),
    ('Eu', 8, 9), ('Gd', 9, 9), ('Tb', 10, 9), ('Dy', 11, 9), ('Ho', 12, 9), ('Er', 13, 9), ('Tm', 14, 9), ('Yb', 15, 9),
    ('Lu', 16, 9), ('Hf', 3, 5), ('Ta', 4, 5), ('W', 5, 5), ('Re', 6, 5), ('Os', 7, 5), ('Ir', 8, 5), ('Pt', 9, 5),
    ('Au', 10, 5), ('Hg', 11, 5), ('Tl', 12, 5), ('Pb', 13, 5), ('Bi', 14, 5), ('Po', 15, 5), ('At', 16, 5), ('Rn', 17, 5),
    # Period 7
    ('Fr', 0, 6), ('Ra', 1, 6), ('Ac', 2, 6), ('Th', 3, 10), ('Pa', 4, 10), ('U', 5, 10), ('Np', 6, 10), ('Pu', 7, 10),
    ('Am', 8, 10), ('Cm', 9, 10), ('Bk', 10, 10), ('Cf', 11, 10), ('Es', 12, 10), ('Fm', 13, 10), ('Md', 14, 10), ('No', 15, 10),
    ('Lr', 16, 10), ('Rf', 3, 6), ('Db', 4, 6), ('Sg', 5, 6), ('Bh', 6, 6), ('Hs', 7, 6), ('Mt', 8, 6), ('Ds', 9, 6),
    ('Rg', 10, 6), ('Cn', 11, 6), ('Nh', 12, 6), ('Fl', 13, 6), ('Mc', 14, 6), ('Lv', 15, 6), ('Ts', 16, 6), ('Og', 17, 6)
]

# Create a custom colormap
colors = ['#FF0000', '#00FF00', '#0000FF']  # Red to Green to Blue
n_bins = 118  # Number of color bins
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10))

el_assoc = {}
# Plot each element
for i, (element, x, y) in enumerate(elements):
    color = cmap(i / (n_bins - 1))
    el_assoc[element] = list(color)


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

        arr = [0.0]*3
        arr = el_assoc[atoms[atom_type]]
        nodes_array.append(arr)

    return nodes_array