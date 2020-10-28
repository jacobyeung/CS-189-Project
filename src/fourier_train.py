import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from cnn import CNN
import numpy as np
import matplotlib.pyplot as plt
import train

"""Functions"""
torch.manual_seed(42)


def train(model, dataloader, recon_set, epoch, len_train_set):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(zip(dataloader, recon_set)), total=len_train_set/dataloader.batch_size):
        data, recon = data[0], data[1]
        data = data.unsqueeze(1)
        recon = recon.unsqueeze(1)
        data = data.reshape(-1, 1, 144, 144)
        recon = recon.reshape(-1, 1, 144, 144).to(device)
        data = data.to(device)
        optimizer.zero_grad()
        recon_true = model.forward(data)
        loss = model.loss(data, recon)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss / len_train_set


def validate(model, dataloader, recon_set, epoch, fpath, len_val_set):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(zip(dataloader, recon_set)), total=len_val_set/dataloader.batch_size):
            data, recon = data[0], data[1]
            data = data.unsqueeze(1)
            recon = recon.unsqueeze(1)
            data = data.reshape(-1, 1, 144, 144)
            recon = recon.reshape(-1, 1, 144, 144).to(device)
            data = data.to(device)
            optimizer.zero_grad()
            recon_true = model.forward(data)
            loss = model.loss(data, recon)
            running_loss += loss.item()
            if i == 0:
                num_rows = min(data.size(0), dataloader.batch_size)
                both = torch.cat((data.view(num_rows, 1, 144, 144)[:5], recon.view(
                    num_rows, 1, 1414, 144)[:5], recon.view(num_rows, 1, 144, 144)[:5]))
                save_image(both.cpu(), "outputs/" +
                           fpath + "/" + str(epoch) + ".png")
    return running_loss / len_val_set


def path_maker(fpath):
    Path("outputs/" + fpath).mkdir(exist_ok=True)


def visualize(fpath, n_samples, train_loss, val_loss):
    figure, axis = plt.subplots(constrained_layout=True)
    axis.plot(np.arange(n_samples), train_loss,
              color="blue", label="Train Loss")
    axis.plot(np.arange(n_samples), val_loss, color="red", label="Val Loss")
    axis.set_xlabel("Number of Training Epochs")
    axis.set_ylabel("Loss")
    figure.legend()
    axis.set_title(fpath + " Loss Visualization.png")
    figure.savefig(fpath + " Loss Visualization.png")


class CustomDataset(Dataset):
    """Planets Dataset"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)


epochs = 15
lr = 0.005
batch_size = 32

"""Start of training script"""
# Hyperparameters
fpaths = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
for fpath in fpaths:
    path_maker(fpath)

    root = "combined_data_matrix/" + fpath + ".npz"
    root_f = "fourier_inputs/" + fpath + ".npz"
    # root = "outputs/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    data = np.load(root)
    data = data['data']
    data_f = np.load(root_f)
    data_f = data_f['data']
    data = torch.from_numpy(data).float()
    data_f = torch.from_numpy(data_f).float()
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

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    train_loader_f = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader_f = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    model = CNN().to(device)
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'vgg11', pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_losses = torch.tensor([0] * epochs)
    v_losses = torch.tensor([0] * epochs)
    for epoch in range(epochs):
        train_loss = train(model, train_loader_f,
                           train_loader, epoch, len_train_set)
        val_loss = validate(model, val_loader_f, val_loader,
                            epoch, fpath, len_val_set)
        t_losses[epoch] = train_loss
        v_losses[epoch] = val_loss
        print("Train Loss: " + str(train_loss))
        print("Val Loss: " + str(val_loss))
    # np.savez(fpath + "_loss", train_loss=train_loss, val_loss=val_loss)
    torch.save(model.state_dict(), "./model_version" + fpath + ".pt")
    visualize(fpath, epochs, t_losses.numpy(), v_losses.numpy())
