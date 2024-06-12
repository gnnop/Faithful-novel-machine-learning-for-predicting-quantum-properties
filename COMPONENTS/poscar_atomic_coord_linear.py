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
        
        atomRep = atomToCoords(atoms[atomType])
        rw = [0.0]*7
        cl = [0.0]*16
        sp = [0.0]*1
        rw[atomRep[0]] = 1.0
        cl[atomRep[1]] = 1.0
        sp[0] = atomRep[2]#The spin is a single bit, so we drop it in one thing
        #We have the whole representation right now - after this I add it into the graph for clarity
        nodesArray.append(unpackLine(poscar[8+i][0:3]))