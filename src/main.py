import numpy as np
import os as os
from PIL import Image


def read_png(file):
    # Outputs a list of numpy matrices read in from png files
    npy_frame = np.load(file)['data']
    return npy_frame


def fourier_transform(matrix):
    # Converts a numpy matrix into its equivalent fourier featurized matrix
    #n = matrix.shape[0]
    #for i in range(0, n):
    #    matrixi = matrix[i, :].reshape(144, 144)
    #    matrix[i, :] = np.fft.fft2(matrixi).reshape(1, 20736)
    #return matrix
    return np.fft.fft2(matrix)


def main():
    # Main script to be runned for program
    file_dir = "C:/Users/Arjun Chandran/Documents/189Project/CS-189-Project/Fourier_transformer/output"
    for i in os.scandir(file_dir):
        name = i.name
        array = read_png(i.path)
        array_fourier = fourier_transform(array)
        file_name = str(name) + "fourier"
        np.savez(file_name, data=array_fourier)
        print("done")

##run file
main()
