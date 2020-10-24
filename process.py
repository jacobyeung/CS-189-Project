import cv2, os, re

def load_metadata(filename):
	assert(os.exists(filename))

	idx = 0
	metadata_dict = {}

	with open(filename) as f:
		location_data = f.readline()
		metadata_dict['location'] = location_data.rstrip().split(' ')
		metadata_dict['body_locs'] = {}

		start_pattern = re.compile(r"^\d+.\d+")
		
		line = f.readline()
		while line:
			if start_pattern.match(line):





def load_image(filename):
	assert(os.exists(filename))


def main():
	parser = argparse.ArgumentParser(description='API GW CI demo toolkit -> ' + os.path.basename(__file__), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument('--data_path', action="store", default=os.environ['HOME'] + "/.edgerc", help="Full or relative path to .edgerc file")
	# parser.add_argument('--section', action="store", default="default", help="The section of the edgerc file with the proper {OPEN} API credentials.")
	parser.add_argument('--data_path', action="store", default="D:/Stellarium/screenshots", help="Datapath to read from.")
	parser.add_argument('--output_path', action="store", default="D:/Stellarium/screenshots/dataset", help="Location for output")
	args = parser.parse_args()

	if not os.exists(args['output_path']):
		os.mkdir(args['output_path'])

	DATA_PATH = args['data_path']
	OUTPUT_PATH = args['output_path']

	metadata_filename = os.path.join(DATA_PATH, 'output.txt')
	metadata = load_metadata(metadata_filename)

if __name__ == '__main__':
	main()
