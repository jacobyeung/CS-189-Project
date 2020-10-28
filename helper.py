import numpy as np
import argparse, os, sys

PLANET_LIST = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

def create_data_matrix_planets(mask_directory, name_directory, output_dir='output'):
	idx = 0

	output_arr = [[] for i in range(len(PLANET_LIST))]

	assert(os.path.exists(mask_directory) and os.path.exists(name_directory))

	while True:
		image_filename = os.path.join(mask_directory, str(idx) + '.npz')
		name_filename = os.path.join(name_directory, str(idx) + '.txt')
		if not os.path.exists(image_filename) or not os.path.exists(name_filename):
			print('done at %s image' % str(idx))
			break

		image_data = np.load(image_filename)['data']
		name_data = open(name_filename).read()

		

		print(image_data.shape)
		print(name_data)

		break

		i += 1


def main():
	create_data_matrix_planets(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
	main()