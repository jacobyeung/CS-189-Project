# CS-189-Project

Early project pulls planet traversal images from Stellarium and contains two data sets. One is a bit mask of the planet. Another is a raw image dataset. We have also added functionalities to generate fourier and convoluted kernel variants of the data.

Stellarium download: https://stellarium.org/

## Instructions

Run `pip install -r requirements.txt` to install all dependencies.

Edit `src/collect_data.scc` for these things:

1. Screenshot output directory, make sure that folder exists if not make one and change the directory appropriately.
2. Choose the number of iterations the code will run for.

Next:

For python scripts, use `python \<script name\> -h` for detailed command-line argument help.

1. Open Stellarium, disable satellite hints and meteor showers
2. Pick a location
3. Pick an FOV
4. F12 to open scripts, load `src/collect_data.ssc`
5. Run `src/collect_data.ssc`, close the scripts menu so it doesn't show up in the image
6. Wait for termination
7. Navigate to %appdata%/Stellarium, get output.txt, place in same folder as images
8. Run `src/process.py`
9. Run `src/helper.py` or `src/featurize_fourier.p`y to augment and featurize the data. `src/helper.py` compiles all the object bit masks into one overall .npz array and has functionality to increase the granularity of the data. `src/featurize_fourier.py` extracts fourier features from the output of src/helper.py.

- For `src/featurize_fourier.py` change the file directory used to your personal directory containing the cleaned.npz files and make sure the data files are in the same directory as `src/featurize_fourier.py`
- Note\*\*\* Due to the way numpy stores complex numbers the fourier version of these files takes up a lot of space (~8GB per file) so be warned.
- The cleaned data in the .npz files can be downloaded from https://drive.google.com/drive/folders/1TFwxn_xk9RVnpkibTUW9kg-ef_b5GC0S?usp=sharing. This data was created using `src/helper.py` with an expand argument of 4.

Training a model:

An example model is included inside this repository, described at the bottom of this README. You can run this model after running `src/process.py` and `src/helper.py` or downloading the cleaned data .npz files in the Google Drive link above. Place them into a folder called "combined_data_matrix" and run `src/train_basic.py`. More detailed instructions are in the section of this README titled **About our Model**.

## The Dataset

<table style="width: 100%;">
  <tr>
    <td width="33%"><img src="https://github.com/jacobyeung/CS-189-Project/blob/main/Raw%20Images/image/0.png"></td>
    <td width="33%"><img src="https://github.com/jacobyeung/CS-189-Project/blob/main/Raw%20Images/object_bit_mask/0.png"></td>
    <td width="33%"><img src="https://github.com/jacobyeung/CS-189-Project/blob/main/Raw%20Images/object_indexed_mask/10000_images/Jupiter.png"></td>
  </tr>
  <tr>
    <td>Raw Image</td>
     <td>Bit Mask Images</td>
     <td>Object Specific Mask Images (Jupiter)</td>
  </tr>
 </table>

Inside `Raw Images/` is an example output from running `src/process.py` on a collection of 25000 images taken from Stellarium. It contains all the outputs described in the write up, shown in the images above, and the provided helper functions can be run on this sample data.

## About Our Model

Tutorial - What is a variational autoencoder?: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

This blog post describes the purpose and construction of variational autoencoders. This helps build intuition for understanding our choice of using the Beta-VAE.

Understanding Disentangling in Beta-VAE: https://arxiv.org/abs/1804.03599

This paper discusses the purpose of adding a beta term in front of the KL divergence term in the loss function of a VAE.

We decided to use a VAE to reconstruct images of the bitmasked data to demonstrate the viability of using our dataset for image to image tasks. We attempted to create smooth and disentangled traversals of the latent distributions as an auxiliary task. (Examples of smooth and disentangled traversals appear in the Understanding Disentangling in Beta-VAE paper.) The Beta-VAE allows us to enforce stricter disentanglement by weighing the KL Divergence term more heavily.

Our specific model works for our bitmasked images - 144x144 pixels and greyscale.

### Instructions to Run Model

1. (Optional) Change `fpaths` in `src/train_basic.py` for planets to train the model on
2. Run `src/train_basic.py`

**Reconstruction Example**

<div style="text-align: center;">
    <img src="https://github.com/jacobyeung/CS-189-Project/blob/main/Reconstruction Examples/Sun/25.png">  
</p>
The images on the top row are the input images; the images on the bottom are the reconstructed images. Each image consists of a white sprite surrounded by black.

**Traversal Example**

<div style="text-align: center;">
    <img src="https://github.com/jacobyeung/CS-189-Project/blob/main/Reconstruction Examples/Sun traversal.png">  
</div>
The images on the top most row are the inputs; the images on the second to top most row are the reconstructions. The rows afterwards represent a traversal of the latent distribution. We think the traversals could be smoother and more disentangled with hyperparameter tuning and differing model complexities.

<br />

See the Early Project Writeup for more details.
