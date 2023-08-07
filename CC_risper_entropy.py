#!/usr/bin/env python 3
"""
The script hereafter implements the following steps:
    - import images from a folder
    - performs the cubical complex filtration based on density thresholding
    - calculate the persistent entropy of the images
    - 
This script is a code written by Samuel Bourgeat,
a Ph.D student in the Jaksic lab at EPFL.
"""

# Import libraries
import time
import glob
import numpy as np
import cripser as cr
import tifffile as TIF
from scipy.stats import entropy
import pandas

print("Packages impoted")

DIRECTORY = "/home/samuel/brainMorpho/Brains_to_CC/" # add your input directory
OUTPUT_DIR = "/home/samuel/TDA/TDA_EntropyHypothesis/" # add your output directory

# Import the tiff files with tifffile

# iterate over files in
# that directory
Files = []
for filename in glob.iglob(f"{DIRECTORY}/*"):
    Files.append(filename)

DGRP_lines = []
Sex = []

# Big iteration over all files

# Define Entropy variable
Entropy_all = []
for file in Files:
    img = TIF.imread(file)
    print("analysing image:", file)
    lines_sex_all = file.split("/")[-1]
    lines_sex = lines_sex_all.split("_")
    DGRP_lines.append(lines_sex[0])
    Sex.append(lines_sex[1].split(".")[0])

    img = img.max() - img
    image = img / img.max()

    print("Initializing images")
    X = image
    # compute PH for the V-construction with the python wrapper (takes time)
    start = time.time()
    pd = cr.computePH(X)
    print(f"elapsed_time:{time.time() - start} sec")

    pds = [pd[pd[:, 0] == i] for i in range(3)]

    XX = [p[:, 1:3] for p in pds]

    # compute the lifespans of each point in the persistence diagram

    lifespan_0 = XX[0][0:-2, 1] - XX[0][0:-2, 0]
    lifespan_1 = XX[1][:, 1] - XX[1][:, 0]
    lifespan_2 = XX[2][:, 1] - XX[2][:, 0]

    Lifespans = [lifespan_0, lifespan_1, lifespan_2]

    entropy_single_brain = []
    for i in Lifespans:
        entropy_single_brain.append(entropy(i, base=2) / np.log2(sum(i))) # Normalized persistence entropy to correct for brain size diversity!

    print("DGRP line:", DGRP_lines[-1], Sex[-1], "Entropy =", entropy_single_brain)
    Entropy_all.append(entropy_single_brain)

df = pandas.DataFrame(Entropy_all, columns=["entropy0", "entropy1", "entropy2"])
df["DGRP"] = DGRP_lines
df["Sex"] = Sex

df.to_csv(OUTPUT_DIR + "Entropy_all_July_20_2023_V4_Normalized_Entropy.csv") # insert the name of the csv fle you want: CHANGE THE VALUE TO FIT YOUR NEEDS
