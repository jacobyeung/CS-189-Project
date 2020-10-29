# CS-189-Project
Early project pulls planet traversal images from Stellarium and contains two data sets. One is a bit mask of the planet. Another is a simple Fourier featurization of the raw pixels.

Instructions to run:

Edit collect_data.scc for these things:

1. Screenshot output directory, make sure that folder exists
2. Number of iterations the code will run for

Instructions

1. Open stellarium, disable satellite hints and meteor showers
2. Pick a location
3. Pick an FOV
4. F12 to open scripts, load collect_data.ssc
5. Run collect_data.ssc, close the scripts menu so it doesn't show up in the image
6. Wait for termination
7. Navigate to %appdata%/Stellarium, get output.txt, place in same folder as images
8. Run process.py with --data_path and --output_path set
9. Run helper.py or src/main.py to augment and featurize the data. Helper.py compiles all the object bit masks into one overall .npz array. src/main.py extracts fourier features from the output of Helper.py.
  - For main.py change the file directory used to your personal directory containing the cleaned.npz files and make sure the data files are in the same directory as main.py
  - Note*** Due to the way numpy stores complex numbers the fourier version of these files takes up a lot of space ~8GB so be warned.
  - The cleaned data in the .npz files can be downloaded from https://drive.google.com/drive/folders/1TFwxn_xk9RVnpkibTUW9kg-ef_b5GC0S?usp=sharing
