import numpy as np
import argparse, os, sys
import cv2

PLANET_LIST = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
RES = (144, 144)

def save_data_matrix(planet, output_masks, output_dir):

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	output_filename = os.path.join(output_dir, planet + '.npz')
	np.savez(output_filename, data=np.array(output_masks))


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
			output_masks.append(np.zeros(RES).flatten())

		idx += 1

	save_data_matrix(planet, output_masks, output_dir)


def main():
	parser = argparse.ArgumentParser(description='Convert planet bitmasks to data matrices', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mask_path', action="store", help="Maskpath to read from.")
	parser.add_argument('--output_path', action="store", default="output", help="Location for output")
	args = parser.parse_args()

	assert(args.mask_path)
	for planet in PLANET_LIST:
		create_data_matrix_planets(planet, args.mask_path, args.output_path)
		print('finished with planet %s' % planet)


if __name__ == '__main__':
	main()