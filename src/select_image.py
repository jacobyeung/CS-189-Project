import numpy as np
import os

root = os.path.abspath(os.getcwd(
) + '/dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
data = np.load(root)
data = data['imgs'][[0, 73728, 147456, 221184, 294912, 36840,
                     442368, 516096, 589824, 663552]]

np.save(os.path.abspath(os.getcwd() + '/src/image.npy'), data)
