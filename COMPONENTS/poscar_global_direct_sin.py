from includes import *


def info(poscar):
    outputGlobalAxisCoordinates = [float(j) for i in poscar[2:5] for j in i.split()]
    # Apply Fourier features to the global axis coordinates
    return [getAbsoluteCoords(i) for i in outputGlobalAxisCoordinates]