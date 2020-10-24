import cv2, os, re, argparse

def load_metadata(filename):
	print(filename)
	assert(os.path.exists(filename))

	idx = -1
	metadata_dict = {}

	with open(filename) as f:
		location_data = f.readline()
		metadata_dict['location'] = location_data.rstrip().split(' ')
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
				metadata_dict['image_data'][idx]['jday'] = float(line.rstrip())

				metadata_dict['image_data'][idx]['planets'] = {}
			else:
				line = line.rstrip().split()
				object_name = line[0]
				obj_pixel_x = int(line[1])
				obj_pixel_y = int(line[2])

				metadata_dict['image_data'][idx]['planets'][object_name] = (obj_pixel_x, obj_pixel_y)

			line = f.readline()


	return metadata_dict
				
def load_image(filename):
	assert(os.path.exists(filename))

def load_images_generate_bitmasks(datapath, outputpath, metadata, prefix='stellarium_ss-'):


def main():
	parser = argparse.ArgumentParser(description='API GW CI demo toolkit -> ' + os.path.basename(__file__), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument('--data_path', action="store", default=os.environ['HOME'] + "/.edgerc", help="Full or relative path to .edgerc file")
	# parser.add_argument('--section', action="store", default="default", help="The section of the edgerc file with the proper {OPEN} API credentials.")
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

	load_images_generate_bitmasks(DATA_PATH, OUTPUT_PATH, metadata)

	print('we did it :)')

if __name__ == '__main__':
	main()
