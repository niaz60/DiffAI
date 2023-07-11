
import numpy as np
import pandas as pd
def f_multi(i, pos_pop, atom_scat, hkl_pos):
    matrix = pos_pop[i, 0] * atom_scat[:, i] * np.exp(2 * np.pi * 1j * hkl_pos[:,i])
    return matrix

def rmv_brkt(string):
    if string == '.':
        return 0
    return float(string.replace("(", "").replace(")", "").replace("-.", "-0.").replace('?', '0').replace("..", "."))


def gaus(x, h):
    const_g = 4 * np.log(2)
    value = ((const_g**(1/2)) / (np.pi**(1/2) * h)) * np.exp(-const_g * (x/h)**2)
    return value

def y_multi(x_val, step, xy_merge, H):
    # y_val = 0
    # xy_idx = 0
    # for xy_idx in range (0, xy_merge.shape[0]):
    #     angle = xy_merge[xy_idx, 0]
    #     inten = xy_merge[xy_idx, 1]
    #     if angle > (x_val * step - 5) and angle < (x_val * step + 5):
    #         y_val = y_val + inten * (gaus((x_val * step - angle), H[xy_idx, 0])*1.5)
    # return y_val
    # simply this function to speed up
    xy_idx = np.arange(0, xy_merge.shape[0])
    angle = xy_merge[xy_idx, 0]
    intern = xy_merge[xy_idx, 1]
    valid_idx = np.where((angle > (x_val * step - 5)) & (angle < (x_val * step + 5)))
    y_val_vector = np.sum(intern[valid_idx] * (gaus((x_val * step - angle[valid_idx]), H[valid_idx, 0])*1.5))
    return y_val_vector


def sym_op(symm_op_line, symm_atom_info):
    
    import numpy as np
    import re
    import pandas as pd   
    
    # Here I use re.split to split the string first(seperated by several signs)
    # Then filter out empty string
    # Then, reversely assign each operations
    symm_op_x = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-3].replace("X", "x").replace("Y", "y").replace("Z", "z")
    symm_op_y = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-2].replace("X", "x").replace("Y", "y").replace("Z", "z")
    symm_op_z = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-1].replace("X", "x").replace("Y", "y").replace("Z", "z")
    
    # Atom positions after apply symmetry operation
    # Here we reduce the shape of matrix from n*6 to n*5, dumped the idx column
    atom_info_symm = np.zeros((symm_atom_info.shape[0], symm_atom_info.shape[1]))
    # We know x, y, z fraction should be copied from previous matrix's column 3, 4, 5
    x = symm_atom_info[:, 2]
    y = symm_atom_info[:, 3]
    z = symm_atom_info[:, 4]
    # Here we use pd.eval to change string to executable expression
    # Inwhich 3 expressions, varaibles are x, y, z defined above
    x_new = pd.eval(symm_op_x)
    y_new = pd.eval(symm_op_y)
    z_new = pd.eval(symm_op_z)
    
    # Build the matrix that stores atom information after applying symmetry op
    # This column is atom number, remain unchanged
    atom_info_symm[:, 0] = symm_atom_info[:, 0]
    atom_info_symm[:, 1] = symm_atom_info[:, 1]
    # These 3 columns are changed, they are coordinates
    atom_info_symm[:, 2] = x_new
    atom_info_symm[:, 3] = y_new
    atom_info_symm[:, 4] = z_new
    # This column is atom occupancy, remain unchanged
    atom_info_symm[:, 5] = symm_atom_info[:, 5]
    
    # Return the newly built matrix, ready to be appended
    return atom_info_symm


def hkl(hkl_max):
    
    import numpy as np
    
    # First we must define a complete set of (hkl) planes. To cover all cases, we need a automatic algorithum to generate the matrix.
    # hkl groups start from 000 to nnn, here n is the upper limit needed to be set. The larger the n, the more planes we are dealing.
    # from n we define hklMax as the first parameters we want.
    
    # we use "vstack" to generate hkl matrix one row by one row.
    # we have hkl_info and hkl_add
    hkl_info = np.array([[1, 0, 0]])
    hkl_add = np.zeros((2, 3))
    # Here we start our loop to increase hkl rows one by one sequentially
    hkl_idx = 0
    hkl_h = 1
    print("Generating hkl_info")
    while hkl_h <= hkl_max:
        if hkl_info[hkl_idx, 1] == hkl_info[hkl_idx, 2] and hkl_info[hkl_idx, 0] != hkl_info[hkl_idx, 1]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 1] = hkl_add[0, 1] + 1
            hkl_add[0, 2] = 0
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 1] > hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 2] = hkl_add[0, 2] + 1
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 1] and hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 0] = hkl_add[0, 0] + 1
            hkl_add[0, 1] = 0
            hkl_add[0, 2] = 0
            if hkl_h != hkl_max:
                hkl_info = np.vstack([hkl_info, hkl_add[0]])
            hkl_h += 1
        hkl_idx += 1
    # Above, "hkl_info" has been calculated
    
    # Then we need to switch hkl positions to garantee 100, 010, 001
    # The process is simple, switch 01, 12, 02, then displace one by one abc -> cab -> bca then reduce identical row
    # First, switch 01, 12, 02
    # print("Generating hkl_exp")
    hkl_exp = np.zeros((1, 3))
    hkl_exp = hkl_info[0, :]
    i = 0
    for i in range (0, hkl_info.shape[0]): 
        hkl_switch01 = np.zeros((2, 3))
        hkl_switch01[0, 0] = hkl_info[i, 1]
        hkl_switch01[0, 1] = hkl_info[i, 0]
        hkl_switch01[0, 2] = hkl_info[i, 2]
        hkl_switch12 = np.zeros((2, 3))
        hkl_switch12[0, 0] = hkl_info[i, 0]
        hkl_switch12[0, 1] = hkl_info[i, 2]
        hkl_switch12[0, 2] = hkl_info[i, 1]
        hkl_switch02 = np.zeros((2, 3))
        hkl_switch02[0, 0] = hkl_info[i, 2]
        hkl_switch02[0, 1] = hkl_info[i, 1]
        hkl_switch02[0, 2] = hkl_info[i, 0]
        hkl_displace201 = np.zeros((2, 3))
        hkl_displace201[0, 0] = hkl_info[i, 2]
        hkl_displace201[0, 1] = hkl_info[i, 0]
        hkl_displace201[0, 2] = hkl_info[i, 1]
        hkl_displace120 = np.zeros((2, 3))
        hkl_displace120[0, 0] = hkl_info[i, 1]
        hkl_displace120[0, 1] = hkl_info[i, 2]
        hkl_displace120[0, 2] = hkl_info[i, 0]
        hkl_exp = np.vstack([hkl_exp, hkl_info[i, :]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch01[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch12[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch02[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace201[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace120[0]])
    
    # Then, reduce identical row
    # print("Generating hkl_redu")
    hkl_redu = np.zeros((1, 3))
    hkl_redu[0] = hkl_exp[0]
    # Loop for extract
    i = 1
    for i in range (1, hkl_exp.shape[0]):
        # Loop for line by line comparasion
        vstack_judge = True
        if_loop_judge = False
        j = 0
        for j in range (0, hkl_redu.shape[0]):
            if np.array_equal(hkl_exp[i], hkl_redu[j]):
                vstack_judge = False
            if_loop_judge = True
        if vstack_judge and if_loop_judge:
            hkl_redu = np.vstack([hkl_redu, hkl_exp[i]])
        # print("vstack_judge\n", vstack_judge, j, "\n")
    
    # Now, we put negetive signs in the matrix
    # for hkl_exp, we extract every line and then vstack to hkl_exp2\
    # print("Generating hkl_exp2")
    i = 0
    hkl_exp2 = np.zeros((1, 3))
    hkl_exp2 = hkl_redu[i, :]
    for i in range (1, hkl_redu.shape[0]): 
        # 1st case: 2 0s
        if hkl_redu[i, 0]*hkl_redu[i, 1] == 0 and  hkl_redu[i, 0]*hkl_redu[i, 2] == 0 and hkl_redu[i, 1]*hkl_redu[i, 2] == 0:
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
        # 2nd case: 1 0s
        elif hkl_redu[i, 0] == 0 or  hkl_redu[i, 2] == 0 or hkl_redu[i, 1] == 0:
            hkl_one0_1 = np.zeros((2, 3))
            if hkl_redu[i, 2] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = -hkl_redu[i, 1]
                hkl_one0_1[0, 2] = hkl_redu[i, 2]
            elif hkl_redu[i, 0] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            elif hkl_redu[i, 1] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_one0_1[0, :]])
        # 3rd case: none 0
        else:
            hkl_none0_1 = np.zeros((2, 3))
            hkl_none0_2 = np.zeros((2, 3))
            hkl_none0_3 = np.zeros((2, 3))
            hkl_none0_1[0, 0] = hkl_redu[i, 0]
            hkl_none0_1[0, 1] = -hkl_redu[i, 1]
            hkl_none0_1[0, 2] = hkl_redu[i, 2]
            hkl_none0_2[0, 0] = hkl_redu[i, 0]
            hkl_none0_2[0, 1] = hkl_redu[i, 1]
            hkl_none0_2[0, 2] = -hkl_redu[i, 2]
            hkl_none0_3[0, 0] = hkl_redu[i, 0]
            hkl_none0_3[0, 1] = -hkl_redu[i, 1]
            hkl_none0_3[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_1[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_2[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_3[0, :]])
            
    # Then we calculate the multiplicity of each hkl planes. The rules are simply, no 0 -> 4, one 0 -> 2, two 0 -> 1
    # print("Generating hkl_multi")
    hkl_multi = np.zeros(( hkl_exp2.shape[0], 1))
    for i in range (0,  hkl_exp2.shape[0]):
        if  hkl_exp2[i, 0] != 0 and  hkl_exp2[i, 1] != 0 and  hkl_exp2[i, 2] != 0:
            hkl_multi[i] = 1
        elif  hkl_exp2[i, 0]* hkl_exp2[i, 1] == 0 and   hkl_exp2[i, 0]* hkl_exp2[i, 2] == 0 and  hkl_exp2[i, 1]* hkl_exp2[i, 2] == 0:
            hkl_multi[i] = 1
        else:
            hkl_multi[i] = 1
    hkl_final = np.hstack([ hkl_exp2, hkl_multi])

    return hkl_final
    