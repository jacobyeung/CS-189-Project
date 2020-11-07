#scripts/helper functions to augment data

import numpy as np
import argparse, os, sys
import cv2
from scipy import ndimage

PLANET_LIST = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
RES = (144, 144)

def save_data_matrix(planet, output_masks, output_dir):

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	output_filename = os.path.join(output_dir, planet + '.npz')
	np.savez(output_filename, data=np.array(output_masks))


#simply append all bit masks for the planet into a big numpy matrix, flattened images
def create_data_matrix_planets(planet, mask_directory, output_dir='output'):
	idx = 0

	output_masks = []

	assert(os.path.exists(mask_directory))

	while True:
		image_dir = os.path.join(mask_directory, str(idx) + '_images')
		if not os.path.isdir(image_dir):
			print('done at %s image' % str(idx))
			break

		image_path = os.path.join(image_dir, planet + '.png')
		if os.path.exists(image_path):
			image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).flatten().astype(dtype=bool)
			output_masks.append(image_data)
		else:
			output_masks.append(np.zeros(RES).flatten().astype(dtype=bool))

		idx += 1

	save_data_matrix(planet, output_masks, output_dir)


#simply append all bit masks for the planet into a big numpy matrix, flattened images
#runs a convolution kernel on each bit mask, kernel is set inside variable k
#current kernel is a simple expansion kernel, expands every point from 1 x 1 to expand_amount x expand_amount
def create_data_matrix_planets_expanded(planet, mask_directory, output_dir='output', expand_amount=4,):
	idx = 0

	k = np.ones((2 * expand_amount + 1, 2 * expand_amount + 1))

	output_masks = []

	assert(os.path.exists(mask_directory))

	while True:
		image_dir = os.path.join(mask_directory, str(idx) + '_images')
		if not os.path.isdir(image_dir):
			print('done at %s image' % str(idx))
			break

		image_path = os.path.join(image_dir, planet + '.png')
		if os.path.exists(image_path):
			#.flatten().astype(dtype=bool)
			image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
			mask = ndimage.convolve(image_data, k, mode='constant', cval=0.0)
			output_masks.append(mask.flatten().astype(dtype=bool))
		else:
			output_masks.append(np.zeros(RES).flatten().astype(dtype=bool))

		idx += 1

	save_data_matrix(planet, output_masks, output_dir)


def main():
	parser = argparse.ArgumentParser(description='Convert planet bitmasks to data matrices', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mask_path', action="store", help="Maskpath to read from.")
	parser.add_argument('--output_path', action="store", default="output", help="Location for output")
	args = parser.parse_args()

	assert(args.mask_path)
	for planet in PLANET_LIST:
		create_data_matrix_planets_expanded(planet, args.mask_path, args.output_path)
		print('finished with planet %s' % planet)


if __name__ == '__main__':
	main()