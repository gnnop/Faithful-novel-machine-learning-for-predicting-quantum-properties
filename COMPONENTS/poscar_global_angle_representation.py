from includes import *


def info(poscar):
    a = unpack_line(poscar[2])
    b = unpack_line(poscar[3])
    c = unpack_line(poscar[4])
    alpha = np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))) / np.pi
    beta = np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))) / np.pi
    gamma = np.arccos(np.dot(b, a) / (np.linalg.norm(b) * np.linalg.norm(a))) / np.pi
    return [a, b, c, alpha, beta, gamma]
