import numpy as np
import os

"""
Selects a subset of images to traverse over.

planet: change to name of solar system object
"""


def select_image(planet):
    planet = "sun"
    root = os.path.abspath(os.getcwd(
    ) + '/combined_matrix_output/' + planet + '.npz')
    data = np.load(root)
    data = data['data'][[0, 2500, 5000, 7500,
                         10000, 12500, 15000, 17500, 20000, 22500]]

    np.save(os.path.abspath(os.getcwd() +
                            '/src/image_sample_' + planet + '.npz'), data)
