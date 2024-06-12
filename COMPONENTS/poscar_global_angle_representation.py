from includes import *

def info(poscar):
    a = unpackLine(poscar[2])
    b = unpackLine(poscar[3])
    c = unpackLine(poscar[4])
    alpha = np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))) / np.pi
    beta = np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))) / np.pi
    gamma = np.arccos(np.dot(b, a) / (np.linalg.norm(b) * np.linalg.norm(a))) / np.pi
    return [a, b, c, alpha, beta, gamma]