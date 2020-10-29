import numpy as np
import os as os


def read_png(file):
    # Outputs a list of numpy matrices read in from png files
    npy_frame = np.load(file)['data']
    return npy_frame


def fourier_transform(matrix):
    # Converts a numpy matrix into its equivalent fourier featurized matrix
    return np.fft.fft2(matrix)


def main():
    # Main script to be runned for program
    file_dir = "C:/Users/zijin/OneDrive/Documents/GitHub/CS-189-Project/src/lost10gbsofspace"
    for i in os.scandir(file_dir):
        name = i.name
        array = read_png(i.path)
        array_fourier = fourier_transform(array)
        file_name = str(name) + "fourier"
        np.savez(file_name, data=array_fourier)
        print("done")

##run file
main()
