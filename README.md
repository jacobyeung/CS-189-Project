# CS-189-Project
Early project pulls planet traversal images from Stellarium and contains two data sets. One is a bit mask of the planet. Another is a simple Fourier featurization of the raw pixels.

Instructions to run:

Edit collect_data.scc for these things:

1. screenshot output directory, make sure that folder exists
2. number of iterations the code will run for

Instructions

1. open stellarium, disable satellite hints and meteor showers
2. pick a location
3. pick an FOV
4. f12 to open scripts, load collect_data.ssc
5. run collect_data.ssc, close the scripts menu so it doesn't show up in the image
6. Wait for termination
7. Navigate to %appdata%/Stellarium, get output.txt, place in same folder as images
8. Run process.py with --data_path and --output_path set
