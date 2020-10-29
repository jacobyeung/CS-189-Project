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

"""Functions"""
torch.manual_seed(42)


def train(model, dataloader, epoch, len_train_set):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len_train_set/dataloader.batch_size):
        data = data.unsqueeze(1)
        data = data.reshape(-1, 1, 144, 144)
        data = data.to(device)
        optimizer.zero_grad()
        recon = model.forward(data)
        loss = model.loss(data, recon)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss / len_train_set


def validate(model, dataloader, epoch, fpath, len_val_set):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len_val_set/dataloader.batch_size):
            data = data.unsqueeze(1)
            data = data.reshape(-1, 1, 144, 144)
            data = data.to(device)
            recon = model.forward(data)
            loss = model.loss(data, recon)
            running_loss += loss.item()
            if i == 0:
                num_rows = min(data.size(0), dataloader.batch_size)
                both = torch.cat((data.view(num_rows, 1, 144, 144)[:5],
                                  recon.view(num_rows, 1, 144, 144)[:5]))
                save_image(both.cpu(), "outputs/" +
                           fpath + "/" + str(epoch) + ".png", nrow=5)
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


# def traverse(model, h_dim, pixels, file_path, number, low, high, personalized):
#     with torch.no_grad():
#         # Input image and change mean, keeping variance the same
#         if pixels.shape[0] == 10:
#             pixels = torch.tensor(
#                 np.reshape([pixels] * 4, (h_dim, 29, 29))).unsqueeze(1).float().to(device)
#         else:
#             pixels = torch.tensor(
#                 [pixels] * h_dim).unsqueeze(1).float().to(device)
#         x = model.encode(pixels)
#         mu, logvar = x[:, :h_dim], x[:, h_dim:]
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         actual = False
#         indices = torch.tensor(np.linspace(low, high, h_dim)).to(device)
#         image = False
#         for i in range(len(indices)):
#             mu_copy = mu.clone()
#             # for c in range(h_dim // 20):
#             #     for j in range(20):
#             #         mu_copy[j + c * 20, j] = indices[i]
#             step = indices[i]
#             for j in range(h_dim):
#                 mu_copy[j, j] = step[j]

#             sample = mu_copy + std * eps

#             sample = model.decode(sample).cpu()
#             sample = sample.view(-1, 1, 29, 29)

#             kld = torch.abs(
#                 (-0.5 * (1 + logvar - mu_copy.pow(2) - logvar.exp()).mean(dim=0)))
#             sorted_kld, indexes = torch.sort(kld, descending=True)
#             sample = sample[indexes]
#             if image is False:
#                 image = sample
#                 actual, _, _ = model.forward(pixels)
#                 actual = actual.view(-1, 1, 29, 29)
#                 pixels = pixels[indexes]
#                 actual = actual[indexes]
#             else:
#                 image = torch.cat((image, sample))
#         both = torch.cat((pixels.view(-1, 1, 29, 29),
#                           actual.view(-1, 1, 29, 29),
#                           image.view(-1, 1, 29, 29).to(device)))
#         save_image(both.cpu(), file_path + ".png", nrow=h_dim)


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
    data = np.load(root)
    data = data['data']
    data = torch.from_numpy(data).float()

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_total = torch.tensor([0] * (epochs * 5))
    train_bce = torch.tensor([0] * (epochs * 5)).to(device)
    train_kld = torch.tensor([0] * (epochs * 5)).to(device)
    val_total = torch.tensor([0] * (epochs * 5))
    val_bce = torch.tensor([0] * (epochs * 5)).to(device)
    val_kld = torch.tensor([0] * (epochs * 5)).to(device)

    for epoch in range(epochs):
        train_loss = train(model, train_loader, epoch, len_train_set)
        val_loss = validate(model, val_loader, epoch, fpath, len_val_set)

        train_total[epoch] = train_epoch_loss
        train_bce[epoch] = bce_train.detach()
        train_kld[epoch] = kl_train.detach()
        val_total[epoch] = val_epoch_loss
        val_bce[epoch] = bce_val.detach()
        val_kld[epoch] = kl_val.detach()

        print("Train Loss: " + str(train_loss))
        print("Val Loss: " + str(val_loss))
    np.savez(fpath + "_loss", train_total=train_total.cpu().numpy(), train_bce=train_bce.cpu().numpy(), train_kld=train_kld.cpu(
    ).numpy(), val_total=val_total.cpu().numpy(), val_bce=val_bce.cpu().numpy(), val_kld=val_kld.cpu().numpy())
    torch.save(model.state_dict(), "./model_version" + fpath + ".pt")
    # visualize(fpath, epochs, t_losses.numpy(), v_losses.numpy())
    # trav(fpath_no_cv, h_dim, noise=False, low=np.array(
    #     [-3] * int(h_dim / 2)), high=np.array([3] * int(h_dim / 2)))
