from pathlib import Path
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image
from csv import writer
import traversal
import visualization
import VAE
import select_image

# Some hyperparameters
lr = 0.0005
batch_size = 64
C = np.arange(26)  # Creates C values to constrain the KL divergence to
C = C/np.log(2)
C = C / 100
epochs = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class CustomDataset(Dataset):
    """DSprites Dataset"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)


def train(model, dataloader, gamma, C, len_train_set):
    model.train()
    running_loss = 0.0
    bce = 0.0
    kl = 0.0
    for i, data in tqdm(enumerate(dataloader),
                        total=int(len_train_set/dataloader.batch_size)):
        data = data.unsqueeze(1)
        data = data.reshape(-1, 1, 144, 144)
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model.forward(data)
        bce_loss, kld, loss = model.final_loss(
            reconstruction, data, mu, logvar, gamma, C)
        running_loss += loss.item()
        bce += bce_loss
        kl += kld
        loss.backward()
        optimizer.step()
    length = len(dataloader.dataset)
    train_loss = running_loss/length
    bce = bce/length
    kl /= length
    return bce, kl, train_loss


def validate(model, dataloader, gamma, c, fpath, epoch, len_val_set):
    model.eval()
    running_loss = 0.0
    bce = 0.0
    kl = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),
                            total=int(len_val_set/dataloader.batch_size)):
            data = data.unsqueeze(1)
            data = data.reshape(-1, 1, 144, 144)
            data = data.to(device)
            reconstruction, mu, logvar = model.forward(data)
            bce_loss, kld, loss = model.final_loss(
                reconstruction, data, mu, logvar, gamma, c)
            bce += bce_loss
            kl += kld
            running_loss += loss.item()

            if i == 0:
                num_rows = min(data.size(0), dataloader.batch_size)
                both = torch.cat((data.view(num_rows, 1, 144, 144)[:5],
                                  reconstruction.view(num_rows, 1, 144, 144)[:5]))
                save_image(both.cpu(), "Reconstruction Examples/" + fpath + "/" + str(epoch) + ".png",
                           nrow=5)

    length = len(dataloader.dataset)
    val_loss = running_loss/length
    bce = bce/length
    kl /= length
    return bce, kl, val_loss


def path_maker(fpath):
    Path("Reconstruction Examples/" + fpath).mkdir(exist_ok=True)


"""
Trains beta VAEs on the below planets.
Main goal is to reconstruct images to show viability of data set for
image to image models.
Auxiliary goal is to develop traversals of the latent distributions.

fpaths: names of planets to reconstruct
"""
fpaths = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
for fpath in fpaths:
    path_maker(fpath)
    root = "combined_data_matrix/" + fpath + ".npz"
    data = np.load(root)
    data = data['data']
    data = torch.from_numpy(data).float()
    batch_size = 32
    dataset = CustomDataset(data)

    total_len = len(data)

    train_set, val_set = random_split(
        dataset, [int(0.95 * total_len), int(0.05 * total_len)])
    print(total_len)
    len_train_set = len(train_set)
    len_val_set = len(val_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    gamma = 0.005
    count = 0
    prev_val_loss = 0
    best_model = 0
    train_total = []
    train_bce = []
    train_kld = []
    val_total = []
    val_bce = []
    val_kld = []
    train_loss = []
    val_loss = []
    model = VAE.VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    path_maker(fpath)
    count = 0
    for c in C:
        losses = []
        for i in range(1):
            bce_train, kl_train, train_epoch_loss = train(
                model, train_loader, gamma, c, len_train_set)
            bce_val, kl_val, val_epoch_loss = validate(
                model, test_loader, gamma, c, fpath, count, len_val_set)
            if count == 0:
                train_total = np.array(train_epoch_loss)
                train_bce = bce_train.detach().cpu().numpy()
                train_kld = kl_train.detach().cpu().numpy()
                val_total = np.array(val_epoch_loss)
                val_bce = bce_val.detach().cpu().numpy()
                val_kld = kl_val.detach().cpu().numpy()
            else:
                train_total = np.append(train_total, train_epoch_loss)
                train_bce = np.append(
                    train_bce, bce_train.detach().cpu().numpy())
                train_kld = np.append(
                    train_kld, kl_train.detach().cpu().numpy())
                val_total = np.append(val_total, val_epoch_loss)
                val_bce = np.append(
                    val_bce, bce_val.detach().cpu().numpy())
                val_kld = np.append(val_kld, kl_val.detach().cpu().numpy())
            val_loss.append(val_epoch_loss)
            print("C: " + str(c) + " Train Loss: " + str(train_epoch_loss))
            print("C: " + str(c) + " Val Loss: " + str(val_epoch_loss))
            count += 1
    np.savez(fpath, train_total=train_total, train_bce=train_bce,
             train_kld=train_kld, val_total=val_total, val_bce=val_bce, val_kld=val_kld)
    torch.save(model.state_dict(),
               "./model_version/" + fpath + ".pt")
    select_image.select_image(fpath)
    traversal.trav(fpath)
    visualization.visualization(fpath)
