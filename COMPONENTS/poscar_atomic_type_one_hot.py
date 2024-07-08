from includes import *

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
        
        arr = [0.0]*118
        arr[mendeleev.element(atoms[atomType]).atomic_number-1] = 1.0
        nodesArray.append(arr)

    return nodesArray