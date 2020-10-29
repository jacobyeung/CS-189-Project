import torch
import torchvision
from torchvision.utils import save_image
import numpy as np
import os
import cnn
root = os.path.abspath(os.getcwd() + '/src/image.npy')
pixels = np.load(root)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(4321)
np.random.seed(4321)
# model = reimp.ReImp().to(device)
# model.load_state_dict(torch.load(
#     './src/model_version/contiguous_C10.pt', map_location=lambda storage, loc: storage))


def traverse(model, pixels, file_path):
    with torch.no_grad():
        # Input image and change mean, keeping variance the same
        if pixels.shape[0] == 10:
            pixels = torch.tensor(
                pixels).unsqueeze(1).float().to(device)
        else:
            pixels = torch.tensor(
                [pixels] * 10).unsqueeze(1).float().to(device)
        x = model.encode(pixels)
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

            sample = model.decode(sample).cpu()
            sample = sample.view(-1, 1, 64, 64)

            kld = torch.abs(
                (-0.5 * (1 + logvar - mu_copy.pow(2) - logvar.exp()).mean(dim=0)))
            sorted_kld, indexes = torch.sort(kld, descending=True)
            # sample = sample[indexes]
            if image is False:
                image = sample
                actual, _, _ = model.forward(pixels)
                actual = actual.view(-1, 1, 64, 64)
                # pixels = pixels[indexes]
                # actual = actual[indexes]
            else:
                image = torch.cat((image, sample))
        both = torch.cat((pixels.view(-1, 1, 64, 64).cpu(),
                          actual.view(-1, 1, 64, 64).cpu(),
                          image.view(-1, 1, 64, 64).cpu()))
        save_image(both.cpu(), file_path + ".png", nrow=10)


def trav(model_name):
    model = cnn.CNN.to(device)
    model.load_state_dict(torch.load(
        './Example_Models/' + model_name + '.pt', map_location=lambda storage, loc: storage))
    traverse(model, pixels[0], "outputs/traversal/" + model_name)
