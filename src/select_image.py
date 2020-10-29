import numpy as np
import os

root = os.path.abspath(os.getcwd(
) + '/combined_matrix_output/Jupiter.npz')
data = np.load(root)
data = data['imgs'][0, 5000, 10000, 15000, 20000, 24999]

np.save(os.path.abspath(os.getcwd() + '/src/image_jupiter.npy'), data)
