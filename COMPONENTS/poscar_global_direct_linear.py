from includes import *


def info(poscar):
    return [float(j) for i in poscar[2:5] for j in i.split()]
