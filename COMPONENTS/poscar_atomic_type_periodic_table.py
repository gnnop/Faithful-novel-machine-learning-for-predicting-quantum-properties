from includes import *

def atomToCoords(name):
    if name == "H":
        return [0, 0, 0]
    elif name == "He":
        return [0, 0, 1]#different, i likey here
    elif name == "Li":
        return [1, 0, 0]
    elif name == "Be":
        return [1, 0, 1]
    elif name == "B":
        return [1, 6, 0]
    elif name == "C":
        return [1, 6, 1]
    elif name == "N":
        return [1, 7, 0]
    elif name == "O":
        return [1, 7, 1]
    elif name == "F":
        return [1, 8, 0]
    elif name == "Ne":
        return [1, 8, 1]
    elif name == "Na":
        return [2, 0, 0]
    elif name == "Mg":
        return [2, 0, 1]
    elif name == "Al":
        return [2, 6, 0]
    elif name == "Si":
        return [2, 6, 1]
    elif name == "P":
        return [2, 7, 0]
    elif name == "S":
        return [2, 7, 1]
    elif name == "Cl":
        return [2, 8, 0]
    elif name == "Ar":
        return [2, 8, 1]
    elif name == "K":
        return [3, 0, 0]
    elif name == "Ca":
        return [3, 0, 1]
    elif name == "Sc":
        return [3, 1, 0]
    elif name == "Ti":
        return [3, 1, 1]
    elif name == "V":
        return [3, 2, 0]
    elif name == "Cr":
        return [3, 2, 1]
    elif name == "Mn":
        return [3, 3, 0]
    elif name == "Fe":
        return [3, 3, 1]
    elif name == "Co":
        return [3, 4, 0]
    elif name == "Ni":
        return [3, 4, 1]
    elif name == "Cu":
        return [3, 5, 0]
    elif name == "Zn":
        return [3, 5, 1]
    elif name == "Ga":
        return [3, 6, 0]
    elif name == "Ge":
        return [3, 6, 1]
    elif name == "As":
        return [3, 7, 0]
    elif name == "Se":
        return [3, 7, 1]
    elif name == "Br":
        return [3, 8, 0]
    elif name == "Kr":
        return [3, 8, 1]
    elif name == "Rb":
        return [4, 0, 0]
    elif name == "Sr":
        return [4, 0, 1]
    elif name == "Y":
        return [4, 1, 0]
    elif name == "Zr":
        return [4, 1, 1]
    elif name == "Nb":
        return [4, 2, 0]
    elif name == "Mo":
        return [4, 2, 1]
    elif name == "Tc":
        return [4, 3, 0]
    elif name == "Ru":
        return [4, 3, 1]
    elif name == "Rh":
        return [4, 4, 0]
    elif name == "Pd":
        return [4, 4, 1]
    elif name == "Ag":
        return [4, 5, 0]
    elif name == "Cd":
        return [4, 5, 1]
    elif name == "In":
        return [4, 6, 0]
    elif name == "Sn":
        return [4, 6, 1]
    elif name == "Sb":
        return [4, 7, 0]
    elif name == "Te":
        return [4, 7, 1]
    elif name == "I":
        return [4, 8, 0]
    elif name == "Xe":
        return [4, 8, 1]
    elif name == "Cs":
        return [5, 0, 0]
    elif name == "Ba":
        return [5, 0, 1]
    elif name == "Lu":
        return [5, 1, 0]
    elif name == "Hf":
        return [5, 1, 1]
    elif name == "Ta":
        return [5, 2, 0]
    elif name == "W":
        return [5, 2, 1]
    elif name == "Re":
        return [5, 3, 0]
    elif name == "Os":
        return [5, 3, 1]
    elif name == "Ir":
        return [5, 4, 0]
    elif name == "Pt":
        return [5, 4, 1]
    elif name == "Au":
        return [5, 5, 0]
    elif name == "Hg":
        return [5, 5, 1]
    elif name == "Tl":
        return [5, 6, 0]
    elif name == "Pb":
        return [5, 6, 1]
    elif name == "Bi":
        return [5, 7, 0]
    elif name == "Po":
        return [5, 7, 1]
    elif name == "At":
        return [5, 8, 0]
    elif name == "Rn":
        return [5, 8, 1]
    elif name == "Fr":
        return [6, 0, 0]
    elif name == "Ra":
        return [6, 0, 1]
    elif name == "Lr":
        return [6, 1, 0]
    elif name == "Rf":
        return [6, 1, 1]
    elif name == "Db":
        return [6, 2, 0]
    elif name == "Sg":
        return [6, 2, 1]
    elif name == "Bh":
        return [6, 3, 0]
    elif name == "Hs":
        return [6, 3, 1]
    elif name == "Mt":
        return [6, 4, 0]
    elif name == "Ds":
        return [6, 4, 1]
    elif name == "Rg":
        return [6, 5, 0]
    elif name == "Cn":
        return [6, 5, 1]
    elif name == "Nh":
        return [6, 6, 0]
    elif name == "Fl":
        return [6, 6, 1]
    elif name == "Mc":
        return [6, 7, 0]
    elif name == "Lv":
        return [6, 7, 1]
    elif name == "Ts":
        return [6, 8, 0]
    elif name == "Og":
        return [6, 8, 1]
    elif name == "La":
        return [5, 9, 0]
    elif name == "Ce":
        return [5, 9, 1]
    elif name == "Pr":
        return [5, 10, 0]
    elif name == "Nd":
        return [5, 10, 1]
    elif name == "Pm":
        return [5, 11, 0]
    elif name == "Sm":
        return [5, 11, 1]
    elif name == "Eu":
        return [5, 12, 0]
    elif name == "Gd":
        return [5, 12, 1]
    elif name == "Tb":
        return [5, 13, 0]
    elif name == "Dy":
        return [5, 13, 1]
    elif name == "Ho":
        return [5, 14, 0]
    elif name == "Er":
        return [5, 14, 1]
    elif name == "Tm":
        return [5, 15, 0]
    elif name == "Yb":
        return [5, 15, 1]
    elif name == "Ac":
        return [6, 9, 0]
    elif name == "Th":
        return [6, 9, 1]
    elif name == "Pa":
        return [6, 10, 0]
    elif name == "U":
        return [6, 10, 1]
    elif name == "Np":
        return [6, 11, 0]
    elif name == "Pu":
        return [6, 11, 1]
    elif name == "Am":
        return [6, 12, 0]
    elif name == "Cm":
        return [6, 12, 1]
    elif name == "Bk":
        return [6, 13, 0]
    elif name == "Cf":
        return [6, 13, 1]
    elif name == "Es":
        return [6, 14, 0]
    elif name == "Fm":
        return [6, 14, 1]
    elif name == "Md":
        return [6, 15, 0]
    elif name == "No":
        return [6, 15, 1]
    else:
        print(name)
        exit()


def info(poscar):
    atoms = poscar[5].split()
    numbs = poscar[6].split()

    total = 0
    for i in range(len(numbs)):
        total+=int(numbs[i])
        numbs[i] = total

    nodesArray = []

    curIndx = 0
    atomType = 0
    for i in range(total):
        curIndx+=1
        if curIndx > numbs[atomType]:
            atomType+=1
        
        atomRep = atomToCoords(atoms[atomType])
        rw = [0.0]*7
        cl = [0.0]*16
        sp = [0.0]*1
        rw[atomRep[0]] = 1.0
        cl[atomRep[1]] = 1.0
        sp[0] = atomRep[2]#The spin is a single bit, so we drop it in one thing
        #We have the whole representation right now - after this I add it into the graph for clarity - the ends of the poscar may have extra data
        nodesArray.append([*cl, *rw, *sp])
    return nodesArray
