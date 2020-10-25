"""
takes in images + output.txt metadata
outputs ->
folders:
	loc.txt, observer location 
	folder1: timestamp
		###.txt, timestamp
	folder2: image
		###.png, new res image
	folder3: object_bit_mask
		###.png, object bit mask * 255 (show up as white)
	folder4: object_visible_bit_mask
		###.png, object visible bit mask * 255 (show up as white)
	folder5: object_indexed_mask
		###.png, object bit mask, but key-d (0 -> no object, 1 -> )
	folder6: object_indexed_visible_mask
		###.png, object bit visible mask, but key-d (0 -> no object, 1 -> )
	folder7: object_index_names
		###.txt, #: name, #: name

"""


import cv2, os, re, argparse
import numpy as np
from utils import *

def save_data(idx, outputpath, jday, image, bit_mask_image, bit_mask_visible_image, bit_mask_indexed_image, bit_mask_indexed_visible_image, planet_lst):
	out1 = os.path.join(outputpath, 'timestamp', str(idx) + '.txt')

	with open(out1, 'w') as f:
		f.write(jday)

	out2 = os.path.join(outputpath, 'image', str(idx) + '.png')
	cv2.imwrite(out2, image)

	out3 = os.path.join(outputpath, 'object_bit_mask', str(idx) + '.png')
	cv2.imwrite(out3, bit_mask_image * 255)

	out4 = os.path.join(outputpath, 'object_visible_bit_mask', str(idx) + '.png')
	cv2.imwrite(out4, bit_mask_visible_image * 255)

	out5 = os.path.join(outputpath, 'object_indexed_mask', str(idx) + '.png')
	cv2.imwrite(out5, bit_mask_indexed_image)

	out6 = os.path.join(outputpath, 'object_indexed_visible_mask', str(idx) + '.png')
	cv2.imwrite(out6, bit_mask_indexed_visible_image)

	out7 = os.path.join(outputpath, 'object_index_names', str(idx) + '.txt')
	with open(out7, 'w') as f:
		for i, planet in enumerate(planet_lst):
			f.write(str(i + 1) + ' ' + planet + '\n')

def load_metadata(filename):
	print(filename)
	assert(os.path.exists(filename))

	idx = -1
	metadata_dict = {}

	with open(filename) as f:
		location_data = f.readline()
		metadata_dict['location'] = location_data.rstrip()

		fov = f.readline()
		metadata_dict['fov'] = fov

		metadata_dict['image_data'] = []

		start_pattern = re.compile(r"^\d+.\d+")
		
		line = f.readline()
		while line:
			#empty line, only \n
			if len(line) == 1:
				pass
			#starting new image
			elif start_pattern.match(line.rstrip()):
				idx += 1
				metadata_dict['image_data'].append({})

				#first line is jday
				metadata_dict['image_data'][idx]['jday'] = line.rstrip()

				metadata_dict['image_data'][idx]['planets'] = {}
			else:
				line = line.rstrip().split()
				object_name = line[0]
				obj_pixel_x = int(line[1])
				obj_pixel_y = int(line[2])

				metadata_dict['image_data'][idx]['planets'][object_name] = (obj_pixel_x, obj_pixel_y)

			line = f.readline()


	return metadata_dict
			
#loads the image at filename
#resizes to res
#returns image, (res_x/orig_res_x, res_y/orig_res_y)	
def load_image_and_resize(filename, res):
	assert(os.path.exists(filename))

	image = cv2.imread(filename)
	image_resized = cv2.resize(image, res, interpolation = cv2.INTER_AREA)

	return image_resized, (image_resized.shape[1] / image.shape[1], image_resized.shape[0] / image.shape[0])

#load images from datapath with this prefix
#omit the prefix from the output (i guess)
def load_images_generate_bitmasks(datapath, outputpath, metadata, prefix='stellarium_ss-', res=(144, 144)):
	assert(os.path.isdir(datapath))

	images = [img for img in os.listdir(datapath) if ('.png' in img and prefix in img)]
	
	idx = 0
	#main loop
	while True:
		image_name = prefix + str(idx).zfill(3) + '.png'

		#done with images
		if image_name not in images:
			break

		image, ratio = load_image_and_resize(os.path.join(datapath, image_name), res)

		# cv2.imshow('test', image)
		# cv2.waitKey(0)

		image_metadata = metadata['image_data'][idx]['planets']
		image_metadata_rescaled = {}
		for planet, loc in image_metadata.items():
			image_metadata_rescaled[planet] = (int(round(loc[0] * 1.0 * ratio[0])), int(round(loc[1] * 1.0 * ratio[1])))

		bit_mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
		bit_mask_visible_image = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
		bit_mask_indexed_image = np.zeros((image.shape[0], image.shape[1]))
		bit_mask_indexed_visible_image = np.zeros((image.shape[0], image.shape[1]))

		planet_idx = 1
		planet_lst = []
		for planet, loc in image_metadata_rescaled.items():

			if loc[0] < 0 or loc[0] >= res[0] or loc[1] < 0 or loc[1] >= res[1]:
				continue
			else:
				fill, fill_visible = flood_fill_bit_vis(loc, image, 1)
				ind_fill, ind_fill_visible = flood_fill_bit_vis(loc, image, planet_idx)
				planet_lst.append(planet)

				bit_mask_image = np.logical_or(bit_mask_image, fill)
				bit_mask_visible_image = np.logical_or(bit_mask_visible_image, fill_visible)

				bit_mask_indexed_image = np.add(bit_mask_indexed_image, ind_fill)
				bit_mask_indexed_visible_image = np.add(bit_mask_indexed_visible_image, ind_fill_visible)

				planet_idx += 1
		
		bit_mask_image = bit_mask_image.astype(np.float64)
		bit_mask_visible_image = bit_mask_visible_image.astype(np.float64)

		save_data(idx, outputpath, metadata['image_data'][idx]['jday'], image, bit_mask_image, bit_mask_visible_image, bit_mask_indexed_image, bit_mask_indexed_visible_image, planet_lst)
		idx += 1

def get_orig_res(datapath, prefix='stellarium_ss-'):
	assert(os.path.isdir(datapath))

	images = [img for img in os.listdir(datapath) if ('.png' in img and prefix in img)]
	img = cv2.imread(os.path.join(datapath, images[0]))
	return img.shape


def save_metadata(output_path, location_data, fov, metadata_orig_res):
	output_file = os.path.join(output_path, 'metadata.txt')
	with open(output_file, 'w') as f:
		f.write('altitude, longitude, latitude\n')
		f.write(location_data + '\n')
		f.write('fov\n')
		f.write(fov)
		f.write('(y, x, channels)\n')
		f.write(str(metadata_orig_res))

def generate_output_folders(output_path):
	if not os.path.isdir(os.path.join(output_path, 'timestamp')):
		os.mkdir(os.path.join(output_path, 'timestamp'))
	if not os.path.isdir(os.path.join(output_path, 'image')):
		os.mkdir(os.path.join(output_path, 'image'))
	if not os.path.isdir(os.path.join(output_path, 'object_bit_mask')):
		os.mkdir(os.path.join(output_path, 'object_bit_mask'))
	if not os.path.isdir(os.path.join(output_path, 'object_visible_bit_mask')):
		os.mkdir(os.path.join(output_path, 'object_visible_bit_mask'))
	if not os.path.isdir(os.path.join(output_path, 'object_indexed_mask')):
		os.mkdir(os.path.join(output_path, 'object_indexed_mask'))
	if not os.path.isdir(os.path.join(output_path, 'object_indexed_visible_mask')):
		os.mkdir(os.path.join(output_path, 'object_indexed_visible_mask'))
	if not os.path.isdir(os.path.join(output_path, 'object_index_names')):
		os.mkdir(os.path.join(output_path, 'object_index_names'))


def main():
	parser = argparse.ArgumentParser(description='Convert Stellarium images to our dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data_path', action="store", help="Datapath to read from.")
	parser.add_argument('--output_path', action="store", help="Location for output")
	args = parser.parse_args()

	assert(args.data_path and args.output_path)

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	DATA_PATH = args.data_path
	OUTPUT_PATH = args.output_path

	metadata_filename = os.path.join(DATA_PATH, 'output.txt')
	metadata = load_metadata(metadata_filename)

	metadata_orig_res = get_orig_res(DATA_PATH)
	save_metadata(OUTPUT_PATH, metadata['location'], metadata['fov'], metadata_orig_res)
	generate_output_folders(OUTPUT_PATH)

	load_images_generate_bitmasks(DATA_PATH, OUTPUT_PATH, metadata)

	print('we did it :)')

if __name__ == '__main__':
	main()
