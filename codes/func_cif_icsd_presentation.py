# This script aims to load cif, seperate cif information, search and find cif information we want
# For XRD we need mainly 3 parts of information:
# 1. CELL: cell information
# 2. SYMM: symmetry operation to expand the whole cell
# 3. ATOM: atom fractional position, occupancy

def cif(cif_dir, cif_file, out_dir, hkl_info, x_min, x_max, cal_mode):
    
    import numpy as np
    import os
    import re
    from IPython.display import display, clear_output
    from func_rmv_brkt import rmv_brkt
    from func_sym_op import sym_op
    from func_hkl import hkl
    from func_peak_shape import gaus, y_multi
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    from func import f_multi
    import time
    
    # This judgement will be used to determine whether CIF is compatible with this code
    cif_readable = True
    
    # Set print options for digits
    # Set 3 decimals
    np.set_printoptions(precision=5)
    # Suppress scientific notations
    np.set_printoptions(suppress=True)
    
# FILE & DICT
# File & dictionary management
    
    # Locate and open CIF files
    cif_file_dir = '{}/{}'.format(cif_dir, cif_file)
    cif_content = open(cif_file_dir)
    cif_content_lines = cif_content.readlines()
    # Open tables for scattering data
    # And create a dictionary using this file
    scat_table = open('chemical table.txt')
    scat_table_lines = scat_table.readlines()
    chem_dict = {}
    for line in scat_table_lines:
        chem_dict[line.split()[0]] = line.split()[1]
     # Create a dictionary mapping space group to crystal structure
    space_group_map_dict = {}
    for i in range(1, 3):
        space_group_map_dict[i] = 1
    for i in range(3, 16):
        space_group_map_dict[i] = 2
    for i in range(16, 75):
        space_group_map_dict[i] = 3
    for i in range(75, 143):
        space_group_map_dict[i] = 4
    for i in range(143, 168):
        space_group_map_dict[i] = 5
    for i in range(168, 195):
        space_group_map_dict[i] = 6
    for i in range(195, 231):
        space_group_map_dict[i] = 7
        
    

# VAR
# Define variables needed during extraction

    # 1. CELL: cell data
    
    cell_a = 0
    cell_b = 0
    cell_c = 0
    cell_alpha = 0
    cell_beta = 0
    cell_gamma = 0
    chem_form = ""
    # This is the "_space_group_IT_number" in CIF file
    space_group = 0
    # Once we have "space_group", this become Ture
    space_group_line_judge = False
    # This is the crystal systems, ranging 1 to 7
    crystal_sys = 0

    # 2. ATOM: atomic data
      
    # This is the line index of the start of atoms lists of a CIF file
    atom_start_line = 0 
    # This is the line index of the end of atoms lists of a CIF file
    atom_end_line = 0   
    # This is the line index of the "loop_", with ATOM info followed
    atom_loop_line = 0
    # Once we locate "_atom_label_line", this becomes False
    atom_loop_line_judge = True
    # This is the line index of the "_atom_label_line"
    atom_label_line = 0
    # This is the line index of the "_atom_site_type_symbol"
    ion_label_line = 0
    # Once we locate "_atom_site_type_symbol", this becomes True. 
    # False indicate the ion label is missing and can be followed by neutral atom scattering table
    # True indicate that ion label is identified and can be followed by ion scattering table
    ion_label_line_judge = False
    # This is the line index of the "_atom_site_fract_x"
    atom_x_line = 0
    # This is the line index of the "_atom_site_fract_y"
    atom_y_line = 0
    # This is the line index of the "_atom_site_fract_z"
    atom_z_line = 0
    # This is the line index of the "_atom_site_occupancy"
    atom_ocp_line = 0
    # Once we locate "_atom_site_occupancy", this becomes True
    atom_ocp_line_judge = False
    # Once we locate "_atom_label_line", this becomes True
    atom_start_line_judge = False
    # Once we locate "atom_start_line", this becomes True
    atom_end_line_judge = False
    # Once we locate "atom_start_line", this becomes True. If we find normal end, this becomes False.
    atom_end_blank_judge = False

    # 3. SYMM: symmetry operations
    
    # This is the line index of the start of symm. op. lists of a CIF file
    symm_start_line = 0
    # This is the line index of the start of symm. op. of a CIF file
    symm_end_line = 0
    # Once we locate "_space_group_symop_operation_xyz" or "_symmetry_equiv_pos_as_xyz", this becomes True
    symm_end_line_judge = False
    
# LOOP CIF
# Read CIF for desired variables
    
    line_count = 0
    for line in cif_content_lines:
        line_count += 1
        
        # 1. CELL
        
        if "_database_code_ICSD" in line:
            icsd_coll_code = line.split()[1]
        elif '_space_group_IT_number' in line:
            space_group = rmv_brkt(line.split()[1])
            space_group_line_judge = True
        elif '_cell_length_a' in line: 
            cell_a = rmv_brkt(line.split()[1])
        elif '_cell_length_b' in line: 
            if len(line.split()) > 2:
                return "Failed: _cell_length_b"
            cell_b = rmv_brkt(line.split()[1])
        elif '_cell_length_c' in line:
            cell_c = rmv_brkt(line.split()[1])
        elif '_cell_angle_alpha' in line:
            cell_alpha = rmv_brkt(line.split()[1]) * np.pi / 180.
        elif '_cell_angle_beta' in line:
            cell_beta = rmv_brkt(line.split()[1]) * np.pi / 180.
        elif '_cell_angle_gamma' in line:
            cell_gamma = rmv_brkt(line.split()[1]) * np.pi / 180.
        elif '_chemical_formula_sum' in line:
            chem_form = re.split(r"['\sB]", line, 1)[1].replace("'", "").strip()
            
        # 2. ATOM
        
        elif 'loop_' in line and atom_loop_line_judge:
            atom_loop_line = line_count            
        # Find the label that all CIF has: "_atom_site_label"
        elif '_atom_site_label' == line.strip():
            atom_label_line = line_count
            atom_start_line_judge = True
            atom_loop_line_judge = False
        # Find the label that indicate ionization: '_atom_site_type_symbol'
        elif '_atom_site_type_symbol' == line.strip():
            ion_label_line = line_count
            ion_label_line_judge = True
        # Find xyz fractions: "_atom_site_fract_x" & "...y" & "...z"
        elif '_atom_site_fract_x' == line.strip():
            atom_x_line = line_count
        elif '_atom_site_fract_y' == line.strip():
            atom_y_line = line_count
        elif '_atom_site_fract_z' == line.strip():
            atom_z_line = line_count
        # Find "_atom_site_occupancy"
        elif '_atom_site_occupancy' == line.strip():
            atom_ocp_line = line_count
            atom_ocp_line_judge = True
        # Find the line doesn't start with "_atom" and this line is the start of atoms' list
        elif ('_atom' not in line) and atom_start_line_judge:
            atom_start_line = line_count
            atom_start_line_judge = False
            atom_end_line_judge = True
            atom_end_blank_judge = True
        # Find the line start with either "loop_" or "#End" or blank. The previous line is the end of atoms
        elif atom_end_line_judge and (('loop_' in line) or ('#End' in line)):
            atom_end_line = line_count - 1
            atom_start_line_judge = False
            atom_end_line_judge = False
            atom_end_blank_judge = False
        # Sometimes the CIF file end with nothing
        if atom_end_blank_judge:
            atom_end_line = line_count
        
        # print(atom_start_line, atom_end_line)
        
        # 3. SYMM
        
        # Find "_space_group_symop_operation_xyz" or "_symmetry_equiv_pos_as_xyz"
        if '_space_group_symop_operation_xyz' in line or '_symmetry_equiv_pos_as_xyz' in line:
            symm_start_line = line_count + 1
            symm_end_line_judge = True
        # Find "loop_" after symm. op.
        if 'loop_' in line.strip() and symm_end_line_judge:
            symm_end_line = line_count - 1
            symm_end_line_judge = False

        # Print varaibles out
    # print("\rCIF done")
        
# CAL VAR
# Calculate variables

    # 2. ATOM
    # If no space group information, we return Failed
    if space_group < 1 or space_group > 230:
        return "Failed: Space group(not between 1-230)"
    if not space_group_line_judge:
        return "Failed: Space Group(no space group id)"
    # Mapping space group and crystal structure
    crystal_sys = space_group_map_dict.get(space_group)
    # If either "atom_start_line" or "atom_end_line" is 0, we return False
    if atom_start_line == 0:
        return "Failed: Atom list(no start find)"
    if atom_end_line == 0:
        return "Failed: Atom list(no end find)"
    # "num_sym_atom": number of non-symmetry atoms
    num_symm_atom = 0
    num_symm_atom = atom_end_line - atom_start_line + 1
    # The column number 6 is defined as "type, idx, x, y, z, ocp". May subject to change
    symm_atom_info = np.zeros((num_symm_atom, 6))
    
    # "count_symm_atom" is the idx for future matrix "symm_atom_info"
    for count_symm_atom, line in enumerate(cif_content_lines[atom_start_line - 1: atom_end_line]):
        # If CIF format is "Al1" which is element + label count
        if re.match('([A-z]+)([0-9]+[+-]*)', line.split()[atom_label_line - atom_loop_line - 1]) != None:
            # "symm_atom_type": CIF format label of chemical name
            symm_atom_type = re.match('([A-z]+)([0-9]+[+-]*)', line.split()[atom_label_line - atom_loop_line - 1]).group(1)
            # "sym_atom_idx": CIF format indices for non-symmetry atoms, not the final sequence of atoms
            symm_atom_idx = re.match('([A-z]+)([0-9]+[+-]*)', line.split()[atom_label_line - atom_loop_line - 1]).group(2)
        # If CIF format is "Al" without index
        elif re.match('([A-z]+)([0-9]+[+-]*)', line.split()[atom_label_line - atom_loop_line - 1]) == None:
            symm_atom_type = line.split()[atom_label_line - atom_loop_line - 1]
            symm_atom_idx = "1"
        # Sometimes, CIF missing ATOM info, like this "? ? ? ?"
        if len(line.split()) < (atom_z_line - atom_loop_line):
            return "Failed: Atom list(list not complete)"
        # For this line, extract x, y, z information
        symm_atom_x = line.replace("'", "").split()[atom_x_line - atom_loop_line - 1]
        symm_atom_y = line.replace("'", "").split()[atom_y_line - atom_loop_line - 1]
        symm_atom_z = line.replace("'", "").split()[atom_z_line - atom_loop_line - 1]
        # For this line, extract occupancy information
        if atom_ocp_line_judge:
            symm_atom_ocp = line.split()[atom_ocp_line - atom_loop_line - 1]
        # If no occupancy in CIF, default to 1
        elif not atom_ocp_line_judge:
            symm_atom_ocp = "1"
        # Here generate matrix "symm_atom_info"
        # Here we search for tables for its corresbonding Z
        if chem_dict.get(symm_atom_type) != None:
            symm_atom_info[count_symm_atom, 0] = int(chem_dict.get(symm_atom_type))
            symm_atom_info[count_symm_atom, 1] = rmv_brkt(symm_atom_idx)
            symm_atom_info[count_symm_atom, 2] = rmv_brkt(symm_atom_x)
            symm_atom_info[count_symm_atom, 3] = rmv_brkt(symm_atom_y)
            symm_atom_info[count_symm_atom, 4] = rmv_brkt(symm_atom_z)
            symm_atom_info[count_symm_atom, 5] = rmv_brkt(symm_atom_ocp)
        # This error case happens when atom label is like "ALM"
        elif chem_dict.get(symm_atom_type) == None:
            return "Failed: Can not match neutrual atom type"
        # Then we update [:, 0] if ion label exists and REPLACE NEUTRAL atom index with ION index    
        if True and ion_label_line_judge and re.match('([A-z]+)([1-9]+[+-]+)', line.split()[ion_label_line - atom_loop_line - 1]) != None and chem_dict.get(line.split()[ion_label_line - atom_loop_line - 1].strip()) != None:
            symm_atom_info[count_symm_atom, 0] = int(chem_dict.get(line.split()[ion_label_line - atom_loop_line - 1].strip()))
        # Here we define a estimated cell total number of atom. To filter out cases with too many to save time.
        num_symm_atom_count = 100
        if num_symm_atom > num_symm_atom_count:
            return "Failed: Estimated atom number exceed limit {} (Estimated total: {})".format(num_symm_atom_count, num_symm_atom)
    print("1/5 Done: ATOM", symm_atom_info)
    
    
    
    # 3. SYMM
    
    if cal_mode == 1:

        # Count how many symmetry operations
        num_symm_op = 0
        num_symm_op = symm_end_line - symm_start_line + 1

        # Summarize information
        lattice_atom_info = np.zeros((0, 6))
        # First consider no symmetry information
        if symm_start_line == 0 or symm_end_line == 0:
            lattice_atom_info = np.vstack([lattice_atom_info, symm_atom_info])     
        # Apply symm. op. on every line of symmetry operations
        for line in cif_content_lines[symm_start_line - 1: symm_end_line]:
            if len(list(filter(None, (re.split(r"[\s,']", line))))) < 3:
                return "Failed: Symmetry operations"
            lattice_atom_info = np.vstack([lattice_atom_info, sym_op(line, symm_atom_info)])       

        # After complete "lattice_atom_info", we need to filter out identical expanded atoms
        # This process is a matrix process that compares all rows of the matrix and erase the rows identical to previous ones
        # Here I define a new matrix that is the result of the reduction "lattice_atom_info_redu"
        lattice_atom_info_redu = np.zeros((1, 6))
        lattice_atom_info_redu[0] = lattice_atom_info[0]
        i = 0
        # Loop for extract
        for i in range (0, lattice_atom_info.shape[0]):
            # Loop for line by line comparasion
            vstack_judge = True
            j = 0
            for j in range (0, lattice_atom_info_redu.shape[0]):
                if np.array_equal(lattice_atom_info[i], lattice_atom_info_redu[j]):
                    vstack_judge = False
            if vstack_judge == True:
                lattice_atom_info_redu = np.vstack([lattice_atom_info_redu, lattice_atom_info[i]])
        # print("lattice_atom_info_redu\n", lattice_atom_info_redu, "\n")
        print("2/5 Done: SYMM")
    
    elif cal_mode == 2:
        print("2/5 Skip: SYMM")
    
    # 4. ATOM + SYMM
    
    if cal_mode == 1:
        
        lattice_atom_info_redu = np.around(lattice_atom_info_redu, decimals=2)
        
        # Next, we expand reduced atom info by +1(if position <= 0)
        # Also, for those fraction equals to 0, through translation we would also get the identical position on nearby latttice but sharing the position. For corner, (0, 0, 0), we have 7 more positions, for (0, 0, z), we have 3 more positions.
        # Here we define a expanded matrix, "lattice_atom_info_redu_exp"
        lattice_atom_info_redu_exp = lattice_atom_info_redu
        i = 1
        for i in range (2, 5):
            j = 0
            for j in range (0, lattice_atom_info_redu_exp.shape[0]):
                # print("lattice_atom_info_redu_exp1\n", lattice_atom_info_redu_exp, i, j, "\n")
                if lattice_atom_info_redu_exp[j, i] <= 0:
                    # Here "add_row" is the extra row that +1.
                    # The reason why this is defined as 2*5 is because 1*5 always have assigned value back to origin matrix, error.
                    add_row = np.zeros((2, 6))
                    add_row[0, :] = lattice_atom_info_redu_exp[j, :]
                    add_row[0, i] = add_row[0, i] + 1
                    # print("add_row\n", add_row, i, "\n")
                    # print("lattice_atom_info_redu_exp2\n", lattice_atom_info_redu_exp, i, j, "\n")
                    lattice_atom_info_redu_exp = np.vstack([lattice_atom_info_redu_exp, add_row[0, :]])
                    # print("lattice_atom_info_redu_exp3\n", lattice_atom_info_redu_exp, i, j, "\n")
                # display(cif_file+" "+"ATOM + SYMM(1)"+str(i)+","+str(j)+"/"+str(lattice_atom_info_redu_exp.shape[0]))
                # clear_output(wait = True)
        # print("lattice_atom_info_redu_exp\n", lattice_atom_info_redu_exp, "\n")

        # Next, we expand reduced atom info by -1(if position >= 1)
        i = 1
        for i in range (2, 5):
            j = 0
            for j in range (0, lattice_atom_info_redu_exp.shape[0]):
                # print("lattice_atom_info_redu_exp1\n", lattice_atom_info_redu_exp, i, j, "\n")
                if lattice_atom_info_redu_exp[j, i] >= 1:
                    # Here "add_row" is the extra row that -1.
                    # The reason why this is defined as 2*4 is because 1*4 always have assigned value back to origin matrix, error.
                    add_row = np.zeros((2, 6))
                    add_row[0] = lattice_atom_info_redu_exp[j]
                    add_row[0, i] = add_row[0, i] - 1
                    # print("add_row\n", add_row, i, "\n")
                    # print("lattice_atom_info_redu_exp2\n", lattice_atom_info_redu_exp, i, j, "\n")
                    lattice_atom_info_redu_exp = np.vstack([lattice_atom_info_redu_exp, add_row[0]])
                    # print("lattice_atom_info_redu_exp3\n", lattice_atom_info_redu_exp, i, j, "\n")
                j += 1
                # display(cif_file+" "+"ATOM + SYMM(2)"+str(i)+","+str(j)+"/"+str(lattice_atom_info_redu_exp.shape[0]))
                # clear_output(wait = True)
            i += 1
        # print("lattice_atom_info_redu_exp\n", lattice_atom_info_redu_exp, "\n")

        # Next, we reduce identical ones and <0 or >1
        lattice_atom_info_redu = np.zeros((1, 6))
        lattice_atom_info_redu[0] = lattice_atom_info_redu_exp[0]
        i = 0
        # Loop for extract
        for i in range (0, lattice_atom_info_redu_exp.shape[0]):
            # Loop for line by line comparasion
            vstack_judge = True
            j = 0
            for j in range (0, lattice_atom_info_redu.shape[0]):
                if np.array_equal(lattice_atom_info_redu_exp[i], lattice_atom_info_redu[j]):
                    vstack_judge = False
                else:
                    k = 1
                    for k in range (2, 5):
                        if lattice_atom_info_redu_exp[i, k] < 0:
                            vstack_judge = False
                        elif lattice_atom_info_redu_exp[i, k] > 1:
                            vstack_judge = False
                        k += 1
                j += 1
            if vstack_judge == True:
                lattice_atom_info_redu = np.vstack([lattice_atom_info_redu, lattice_atom_info_redu_exp[i]])
            i += 1
        # print("lattice_atom_info_redu\n", lattice_atom_info_redu, "\n")

        # Next, we rename a simplier matrix
        cell_info = lattice_atom_info_redu
        # print("cell_info\n", cell_info, "\n")
        print("3/5 Done: ATOM + SYMM")

#!!!TEST
        i = 0
        for i in range (0, cell_info.shape[0]):
            if True and cell_info[i, 0] == 7 and cell_info[i, 1] == 1:
                print(cell_info[i, :])
    
    elif cal_mode == 2:
        print("3/5 Skip: ATOM + SYMM")
    
    # PEAK POSITIONS 2THETA

    if cal_mode == 1:
    
        # Since we have hkl_info inputed, we use general equations to calculate distance between hkl planes
        hkl_h = hkl_info[:, 0]
        hkl_k = hkl_info[:, 1]
        hkl_l = hkl_info[:, 2]
        # The general equation is described via Triclinic. Although other would be simplier, but we don't necessarily need to classify.
        a = cell_a
        b = cell_b
        c = cell_c
        alpha = cell_alpha
        beta  = cell_beta
        gamma = cell_gamma
        v = 0
        v = (a*b*c) * ( 1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 )**(1/2)
        hkl_d = np.zeros((hkl_info.shape[0], 1))
        hkl_d = (1/v) * (hkl_h**2*b**2*c**2*np.sin(alpha)**2 + hkl_k**2*a**2*c**2*np.sin(beta)**2 + hkl_l**2*a**2*b**2*np.sin(gamma)**2 + 2*hkl_h*hkl_k*a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma)) + 2*hkl_k*hkl_l*a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha)) + 2*hkl_h*hkl_l*a*b**2*c*(np.cos(alpha)*np.cos(gamma)-np.cos(beta)))**(1/2)
        hkl_d = 1/hkl_d
        
        print("hkl_d")
    
        # Then calculate two_theta
        wavelength   = 1.5418
        two_theta    = np.zeros((hkl_info.shape[0], 1))
        two_theta_pi = np.zeros((hkl_info.shape[0], 1))
        for i in range (0, hkl_info.shape[0]):    
            if wavelength / 2 / hkl_d[i] < 1:
                theta_cal = np.arcsin(wavelength / 2 / hkl_d[i])
                # Here we have the options to add 2theta_errors
                theta_err = 0
                theta_obs = theta_cal + theta_err
                two_theta[i] = 2*theta_obs/np.pi*180
                two_theta_pi[i] = 2*theta_obs
        two_theta = np.around(two_theta, decimals=2)
                
        # Here before hkl matrix is passed on, we delete those hkls, with 0 as 2 theta.
        # hkl_2theta is a n*5 array
        hkl_2theta = np.hstack([hkl_info, two_theta])
        hkl_2theta = np.hstack([hkl_2theta, two_theta_pi])
        print("hkl_2theta")
        # Then we delete its row if 2theta is 0
        i = 0
        loop_judge = True
        while loop_judge:
            if hkl_2theta[i, 4].astype(int) == 0:
                # print("hkl ", i, "th row deleted: ", hkl_2theta[i])
                hkl_2theta = np.delete(hkl_2theta, i, 0)
                i -= 1
            if i == hkl_2theta.shape[0] - 1:
                loop_judge = False
            i += 1
        # print("hkl_2theta\n", hkl_2theta, "\n")
        hkl_info = np.zeros((hkl_2theta.shape[0], 4))
        two_theta = np.zeros((hkl_2theta.shape[0], 1))
        two_theta_pi = np.zeros((hkl_2theta.shape[0], 1))
        hkl_info[:, 0:4] = hkl_2theta[:, 0:4]
        two_theta[:, 0] = hkl_2theta[:, 4]
        two_theta_pi[:, 0] = hkl_2theta[:, 5]
        
        # After two_theta, we calculate lorentz-polarization factor
        lp = np.zeros((hkl_2theta.shape[0], 1))
        for i in range (0, hkl_2theta.shape[0]):
            lp[i] = (1 + np.cos(two_theta_pi[i])**2) / (np.cos(two_theta_pi[i]/2)*np.sin(two_theta_pi[i]/2)**2)
            # lp[i] = (1 + np.cos(two_theta_pi[i])**2) / (np.sin(two_theta_pi[i]))
        print("lp")

    # STRUCTURE FACTOR

        # Next, vector product of h * x_j
        hkl_pos = np.matmul(hkl_info[:, [0, 1, 2]], cell_info[:, [2, 3, 4]].T)
        print("hkl_pos")

        # Next, population factor
        # This depend on the position of the atoms, if one 0/1 -> 1/2, two 0/1 -> 1/4, three 0/1 -> 1/8
        # This population should be determined by symetry, so it is WRONG for now
        pos_pop = np.zeros((cell_info.shape[0], 1))
        i = 0
        for i in range (0, cell_info.shape[0]):
            j = 1
            count = 0
            for j in range (2, 5):
                if cell_info[i, j] == 1 or cell_info[i, j] == 0:
                    count += 1
            if count == 0:
                pos_pop[i, 0] = 1
            elif count == 1:
                pos_pop[i, 0] = 1/2
            elif count == 2:
                pos_pop[i, 0] = 1/4
            elif count == 3:
                pos_pop[i, 0] = 1/8
        print("pos_pop", pos_pop.T)
            
        # Next, temperature factor, we use the simplest equation. b ranges from 0.5-1 or 1-3.
        temp_factor_b = 0
        s = np.zeros((two_theta_pi.shape[0], 1))
        s = np.sin(two_theta_pi/2) / wavelength
        temp_factor = np.exp(-temp_factor_b * s**2)
        print("temp_factor")
   
        
### ATOM SCATTERING FACTOR       
        # Next, atmoic scattering factor. 
        # For neutral atom: International Tables for Crystallography (2006). Vol. C. ch. 4.3, p. 262
        # https://it.iucr.org/Cb/ch4o3v0001/sec4o3o2/
        # For ion: International Tables for Crystallography (2006). Vol. C. ch. 6.1, pp. 554-590
        # https://it.iucr.org/Cb/ch6o1v0001/sec6o1o1/
        atom_scat = np.zeros((hkl_info.shape[0], cell_info.shape[0]))
        # i is the sequense of atoms
        i = 0
        for i in range (0, cell_info.shape[0]):
            col = np.zeros((hkl_info.shape[0], 1))
            abc_ion = np.zeros((9, 1))
            abc_ion[0:9, 0] = np.array(scat_table_lines[int(cell_info[i, 0]) - 1].split()[3:12])
            # j is the appraoch for integration   
            j = 0
            for j in range (0, 7, 2):
                col = col + abc_ion[j] * np.exp(- abc_ion[j+1] * s**2)
            col = col + abc_ion[8]
            # Then multiply by occupancy
            col = col * cell_info[i, 5]
            # here replace the correct result in to matrix 'atom_scat'
            atom_scat[:, i] = col[:, 0]
        print("atom_scat")
       
        # Final equation. This is the loop of n atoms, calculating f_hkl integration. f_hkl is a 2x1 matrix.
        # This calculation takes time!!
        pool = mp.Pool(2)
        f_hkl = pool.starmap(f_multi, [(i, pos_pop, atom_scat, hkl_pos) for i in range (0, cell_info.shape[0])])
        pool.close()
        f_hkl = np.array(f_hkl)
        i = 0
        f_hkl_sum = np.zeros((1, hkl_info.shape[0]))
        for i in range (0, cell_info.shape[0]):
            f_hkl_sum = f_hkl_sum + f_hkl[i, :]
        struc = np.square(np.absolute(f_hkl_sum)).T
        print("struc")
        #f_hkl = np.zeros((hkl_info.shape[0], 1))
        #for i in range (0, cell_info.shape[0]):
        #    f_hkl = f_hkl + pos_pop[i, 0] * atom_scat[:, i] * np.exp(2 * np.pi * 1j * hkl_pos[:,i])
        #struc = np.zeros((hkl_info.shape[0], 1))
        #struc = np.square(np.absolute(f_hkl[0, :].T))
        
    # INTENSITIES
    # Intensities take structural factor, polarization factor, angular-velocity and etc together
        inte = np.zeros((hkl_info.shape[0], 1))
        inte = lp[:, 0] * struc[:, 0]
        x_y = np.zeros((hkl_info.shape[0], 2))
        x_y[:,0] = two_theta[:, 0]
        x_y[:,1] = inte[:]
        xy_hkl = np.around(np.hstack([x_y, hkl_info]), decimals=2)
        print("xy_hkl")
        
    # After having all twotheta and its intensities, we realize that some intensities are zero, we first filter them out.
        xy_redu = np.zeros((0, 2))
        i = 0
        for i in range (0, x_y.shape[0]):
            if x_y[i, 1].astype(int) != 0:
                xy_redu = np.vstack((xy_redu, x_y[i, :]))

    # After having a pure xy data, we merge their intensites if they are in one position
        xy_merge = np.zeros((1, 2))
        xy_merge[0, :] = xy_redu[0, :]
        i = 1
        for i in range (1, xy_redu.shape[0]):
            merge_judge = True
            j = 0
            for j in range (0, xy_merge.shape[0]):
                if xy_redu[i, 0] == xy_merge[j, 0]:
                    xy_merge[j, 1] = xy_merge[j, 1] + xy_redu[i, 1]
                    merge_judge = False
                    break
            if merge_judge:
                xy_merge = np.vstack((xy_merge, xy_redu[i, :]))
        print("xy_merge\n", xy_merge[0: 10])
        
        new_filename = "xy_icsd_{}.txt".format(re.split(r"[.]", icsd_coll_code)[0])
        new_file= open("{}/{}".format(out_dir, new_filename), "w+")
        #np.savetxt(new_file, pattern, delimiter=',')
        new_file.write("\n".join(str(item).replace("[", "").replace("]", "") for item in xy_hkl.tolist()))
        
    # Plot bar
        if False:
            plt.figure(figsize=(15,4))    
            plt.xlim([x_min, x_max])
            plt.bar(x_y[:,0], x_y[:,1], width=0.2, bottom=None, align='center')
            plt.show()
            plt.figure(figsize=(15,8))    
            plt.xlim([x_min, x_max])
            plt.bar(xy_merge[:,0], xy_merge[:,1], width=0.2, bottom=None, align='center', color='red')
            plt.show()
        print("4/5 Done: x_y")
        
    elif cal_mode == 2:
        print("4/5 Skip: x_y")
    
    # Peak Shape Functions
    # Defined in "func_peak_shape.py"
    
    # Set up x-axis and resolutions
    if cal_mode == 1:
        
        U = 0.05
        V = -0.04
        W = 0.03
        H = np.zeros((xy_merge.shape[0], 1))
        H[:, 0] = (U * (np.tan(xy_merge[:, 0]*(np.pi/180)/2))**2 + V * np.tan(xy_merge[:, 0] * (np.pi/180)/2) + W)**(1/2)
        print("5/5 Drawing pattern")
        
        total_points = 18000
        step = 180/total_points
        # Set up a x-y 1D data
        if False:
            time_start = time.time()
            pattern1  = np.zeros((total_points,2))
            # Iterate every x position(2theta) using gaussian function
            x_val = 0
            for x_val in range (0, total_points):
                y_val = 0
                for xyhkl_idx in range (0, xy_merge.shape[0]):
                    if xy_merge[xyhkl_idx, 0] > (x_val * step - 10) and xy_merge[xyhkl_idx, 0] < (x_val * step + 10):
                        y_val = y_val + xy_merge[xyhkl_idx, 1] * (gaus((x_val*step - xy_merge[xyhkl_idx, 0]), H[xyhkl_idx, 0]) + 0.5 * gaus((x_val*step - xy_merge[xyhkl_idx, 0]), H[xyhkl_idx, 0]))
                pattern1[x_val, 0] = x_val*step
                pattern1[x_val, 1] = y_val
            print('pattern1', format(time.time() - time_start, '.3f'))

#multiprocess of peak shape function
        time_start = time.time()    
        pool = mp.Pool(8)
        pattern = pool.starmap(y_multi, [(x_val, step, xy_merge, H) for x_val in range (0, total_points)])
        pool.close()
        pattern2 = np.zeros((total_points,2))
        pattern2[:, 1] = np.asarray(pattern)
        pattern2[:, 0] = np.arange(0,180,step)
        print('pattern2', format(time.time() - time_start, '.3f'))
        
        # Normalization, leaving only 2 dicimal
        if False:
            pattern1[:,0] = pattern1[:,0].round(decimals=2)
            pattern1[:,1] = (pattern1[:,1] / np.max(pattern1[:,1])).round(decimals=3)
        if True:
            pattern2[:,0] = pattern2[:,0].round(decimals=2)
            pattern2[:,1] = (pattern2[:,1] / np.max(pattern2[:,1])).round(decimals=3)
            print('Normalization')

        # Print the plot for preview
        
        if False:
            plt.figure(figsize=(15,4))    
            plt.xlim([x_min, x_max])
            plt.plot(pattern1[:, 0], pattern1[:,1]) 
            plt.show()
        plt.figure(figsize=(15,4))    
        plt.xlim([x_min, x_max])
        plt.plot(pattern2[:, 0], pattern2[:,1]) 
        plt.show()
        # print ("formula = ", chem_form, "\n", "a = ", cell_a, "\n", "b = ", cell_b, "\n", "c = ", cell_c, "\n", "alpha = ", cell_alpha, "\n", "beta  = ",  cell_beta, "\n", "gamma = ", cell_gamma, "\n")

        # Write to new txt files
        print("Writing output")
        new_filename = "icsd_{}.txt".format(re.split(r"[.]", icsd_coll_code)[0])
        new_file= open("{}/{}".format(out_dir, new_filename), "w+")
        #np.savetxt(new_file, pattern, delimiter=',')
        new_file.write(chem_form  + "\n")
        new_file.write("crystal_structure " + str(crystal_sys) + "\n")
        new_file.write("space_group " + str(space_group) + "\n")
        new_file.write("\n".join(str(item).replace("[", "").replace("]", "") for item in pattern2.tolist()))
    
    elif cal_mode == 2:
        print("5/5 Skip: Pattern")
    
    # Close opened files
    cif_content.close()
    scat_table.close()
    if cal_mode == 1:
        new_file.close()
    return "GOOD!"
    
    