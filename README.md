
# CS-189-Project
Early project pulls planet traversal images from Stellarium and contains two data sets. One is a bit mask of the planet. Another is a raw image dataset. We have also added functionalities to generate fourier and convoluted kernel variants of the data.

Stellarium download: https://stellarium.org/

Instructions to run:

Edit src/collect_data.scc for these things:

1. Screenshot output directory, make sure that folder exists if not make one and change the directory appropriately.
2. Choose the number of iterations the code will run for.


Instructions

1. Open Stellarium, disable satellite hints and meteor showers
2. Pick a location
3. Pick an FOV
4. F12 to open scripts, load src/collect_data.ssc
5. Run src/collect_data.ssc, close the scripts menu so it doesn't show up in the image
6. Wait for termination
7. Navigate to %appdata%/Stellarium, get output.txt, place in same folder as images
8. Run src/process.py with --data_path and --output_path command line flags
9. Run src/helper.py or src/featurize_fourier.py to augment and featurize the data. src/helper.py compiles all the object bit masks into one overall .npz array and has functionality to increase the granularity of the data. src/featurize_fourier.py extracts fourier features from the output of src/helper.py.
  - For src/featurize_fourier.py change the file directory used to your personal directory containing the cleaned.npz files and make sure the data files are in the same directory as src/featurize_fourier.py
  - Note*** Due to the way numpy stores complex numbers the fourier version of these files takes up a lot of space (~8GB per file) so be warned.
  - The cleaned data in the .npz files can be downloaded from https://drive.google.com/drive/folders/1TFwxn_xk9RVnpkibTUW9kg-ef_b5GC0S?usp=sharing. This data was created using src/helper.py with an expand argument of 4.

See Early Project Writeup for more details. 
