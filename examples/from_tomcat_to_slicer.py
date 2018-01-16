import numpy as np
import scipy
import astra
import tomop
import transforms3d as tf
import h5py

import sys

# I. convert from TOMCAT hdf5 to ASTRA geometry
# a. read data

if len(sys.argv) == 1:
    print("USAGE: python", sys.argv[0], "<path to data>")
    exit(-1)

path = sys.argv[1]

