import numpy as np

def gaus(x, h):
    const_g = 4 * np.log(2)
    value = ((const_g**(1/2)) / (np.pi**(1/2) * h)) * np.exp(-const_g * (x/h)**2)
    return value

def y_multi(x_val, step, xy_merge, H):
    y_val = 0
    xy_idx = 0
    for xy_idx in range (0, xy_merge.shape[0]):
        angle = xy_merge[xy_idx, 0]
        inten = xy_merge[xy_idx, 1]
        if angle > (x_val * step - 5) and angle < (x_val * step + 5):
            y_val = y_val + inten * (gaus((x_val * step - angle), H[xy_idx, 0])*1.5)
    return y_val
