import numpy as np
import pickle as pickle
import os as os
from PIL import Image
import numpy.random as random


def read_png(file):
    # Outputs a list of numpy matrices read in from png files
    img_frame = Image.open(file)
    npy_frame = np.array(img_frame)
    return npy_frame


def fourier_transform(matrix):
    # Converts a numpy matrix into its equivalent fourier featurized matrix
    rand_key = random.default_rng(10)
    embedding_size = 144
    bvals = random.normal(rand_key, (embedding_size, 2))
    avals = np.ones((bvals.shape[0]))
    return np.concatenate([avals * np.sin((2. * np.pi * matrix) @ bvals.T),
                           avals * np.cos((2. * np.pi * matrix) @ bvals.T)], axis=-1)


def main():
    # Main script to be runned for program
    file_dir = "~CS189Project/Pictures"
    for i in os.scandir(file_dir):
        name = i.name
        array = read_png(i)
        array_fourier = fourier_transform(array)
        file_name = str(name ) + ".obj"
        with open(file_name, "wb") as f:
            pickle.dump(array_fourier, f)
