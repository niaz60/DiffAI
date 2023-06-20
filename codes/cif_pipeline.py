import os
import time
#Astrick import fucntions
from cif_reader import *
from func_hkl import *



def pipieline(in_dir,out_dir, hkl_max, U, V, W):

    if not os.path.exists(out_dir+"/"):
            os.makedirs(out_dir+"/")

    hkl_info = hkl(hkl_max)
    cif_count = 0
    for path, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_count += 1

    # Calculate XRD and show progress
    x_step = 0.01
    cif_cal_count = 0



    for path, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.cif'):
                # Write a log file
                print("Calculating " + path + ": " + file)
                print("Progress: ", str(cif_cal_count), "/", str(cif_count))
                # Here combines cwd and subdirectory
                full_dir = path
                # fileDir = "{}/{}.txt".format(out_dir, re.split(r"[.]", file)[0])
                # Here record timing
                time_start = time.time()
                # Check existing XRD output files in output folder to avoid repeated calculations.
                cif_return = cif(full_dir, file, out_dir, hkl_info,  x_step,  U, V, W )
                # Here record timing
                # time_cost = format(time.time() - time_start, '.3f')
                cif_cal_count += 1
                
    
    # output folder
    config2Theta = "HighRes2Theta_5to90"
    outDir = f"database_datasets/{config2Theta}/ExampleSet/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # output: one .csv file for 7-way labels
    labels7Dir = outDir + "labels7.csv"
    # output: one .csv file for 230-way labels
    labels230Dir = outDir + "labels230.csv"
    # output: one .csv file for features
    featuresDir = outDir + "features.csv"
    # output: one file for ICSD ids
    idsDir = outDir + "ids.csv"

    # create output files
    with open(labels7Dir, "w") as labels7:
        pass
    with open(labels230Dir, "w") as labels230:
        pass
    with open(featuresDir, "w") as features:
        pass
    with open(idsDir, "w") as ids:
        pass

    #Creat List of CIFs
    ListCIFs = os.listdir(out_dir)

    count = 0
    failCount = 0
    # main loop
    for entry in ListCIFs:
        # convert cleaned index to pattern index
        fileName = entry.strip("[\n']").replace(".cif", ".txt")
        # locate the pattern
        fileDir = out_dir + "/" + fileName
        count += 1
        print(f"{fileName} count:{count} failCount:{failCount}")
        # cleaned index might have cifs that fails to calculate XRD
        # check if corresponding xrd exist
        if os.path.isfile(fileDir):
            # read the pattern file
            with open(fileDir) as xrdFile:
                xrdLines = xrdFile.readlines()
            # convert pattern file header to 7-ways
            with open(labels7Dir, "a") as labels7:
                labels7Array = np.zeros((1, 7))
                # 2nd line of pattern file is the 7-way label
                labels7Array[0, int(xrdLines[1].split()[1]) - 1] = 1
                labels7Array = labels7Array.astype(int)
                # save array
                np.savetxt(labels7, labels7Array, fmt="%d", delimiter=",")
            with open(labels230Dir, "a") as labels230:
                labels230Array = np.zeros((1, 230))
                labels230Array[0, int(xrdLines[2].split()[1]) - 1] = 1
                labels230Array = labels230Array.astype(int)
                # save array
                np.savetxt(labels230, labels230Array, fmt="%d", delimiter=",")
    
            with open(featuresDir, "a") as features:
                featuresVector = np.zeros((1, 8500))
                i = 0
                for i in range (0, 8500):
                    featuresVector[0, i] = float(xrdLines[i+3+500].split()[1]) * 1000
                featuresVector = featuresVector.astype(int)
                # save array
                np.savetxt(features, featuresVector, fmt="%d", delimiter=",")
            with open(idsDir, "a") as ids:
                ids.write(fileName.replace(".txt", "").strip())
                ids.write("\n")

if __name__ == "__main__":
    in_dir = "../CIFs_examples"
    out_dir = "../XRD_output_examples"
    hkl_max = 10
    U = 1
    V = 1
    W = 1
    pipieline(in_dir, out_dir, hkl_max, U, V, W)