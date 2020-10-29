import numpy as np
import os

root = os.path.abspath(os.getcwd(
) + '/combined_matrix_output/Sun.npz')
data = np.load(root)
data = data['imgs'][0, 2500, 5000, 7500,
                    10000, 12500, 15000, 17500, 20000, 22500]

np.save(os.path.abspath(os.getcwd() + '/src/image_sample_sun.npy'), data)
