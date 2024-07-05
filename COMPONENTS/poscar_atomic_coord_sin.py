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

        nodesArray.append(flatten([getRelativeCoordinates(i) for i in unpackLine(poscar[8+i])]))
    
    return nodesArray
