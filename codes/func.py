import numpy as np
def f_multi(i, pos_pop, atom_scat, hkl_pos):
    matrix = pos_pop[i, 0] * atom_scat[:, i] * np.exp(2 * np.pi * 1j * hkl_pos[:,i])
    return matrix