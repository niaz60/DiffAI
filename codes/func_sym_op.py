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

        