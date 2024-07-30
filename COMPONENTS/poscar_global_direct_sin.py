from includes import *


def info(poscar):
    output_global_axis_coordinates = [float(j) for i in poscar[2:5] for j in i.split()]
    # apply fourier features to the global axis coordinates
    return flatten([get_absolute_coords(i) for i in output_global_axis_coordinates])
