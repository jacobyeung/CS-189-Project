import torch
import torchvision
from torchvision.utils import save_image
import numpy as np
import os
import VAE

"""
Creates traversal of latent space by linearly changing the mean of latent distributions
across a preset range of values -3 to 3.
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE.VAE().to(device)

torch.manual_seed(4321)
np.random.seed(4321)


def traverse(model, pixels, file_path):
    with torch.no_grad():
        # Input image and change mean, keeping variance the same
        if pixels.shape[0] == 10:
            pixels = torch.tensor(
                pixels).unsqueeze(1).float().to(device)
        else:
            pixels = torch.tensor(
                [pixels] * 10).unsqueeze(1).float().to(device)
        pixels = pixels.reshape(-1, 1, 144, 144)
        x = model.downsize(pixels)
        x = x.view(-1, 5 * 5 * 32)
        x = model.linear(x)
        mu, logvar = x[:, :10], x[:, 10:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        actual = False
        indices = np.arange(-3, 3.1, 0.6)
        image = False
        for i in range(len(indices)):
            mu_copy = mu.clone()
            for j in range(10):
                mu_copy[j, j] = indices[i]
            sample = mu_copy + std * eps

            sample = model.declin_layer(sample)
            sample = sample.view(-1, 32, 5, 5)
            sample = model.upsize(sample).cpu()
            sample = sample.view(-1, 1, 144, 144)

            kld = torch.abs(
                (-0.5 * (1 + logvar - mu_copy.pow(2) - logvar.exp()).mean(dim=0)))
            sorted_kld, indexes = torch.sort(kld, descending=True)
            sample = sample[indexes]
            if image is False:
                image = sample
                actual, _, _ = model.forward(pixels)
                actual = actual.view(-1, 1, 144, 144)
                pixels = pixels[indexes]
                actual = actual[indexes]
            else:
                image = torch.cat((image, sample))
        both = torch.cat((pixels.view(-1, 1, 144, 144).cpu(),
                          actual.view(-1, 1, 144, 144).cpu(),
                          image.view(-1, 1, 144, 144).cpu()))
        save_image(both.cpu(), file_path + ".png", nrow=10)


def trav(model_name):
    root = os.path.abspath(
        os.getcwd() + '/Reconstruction Examples/image_sample_' + model_name + '.npy')
    pixels = np.load(root)

    model.load_state_dict(torch.load(
        './Example Models/' + model_name + '.pt', map_location=lambda storage, loc: storage))
    traverse(model, pixels, "Reconstruction Examples/" +
             model_name + " traversal")
