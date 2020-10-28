import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import save_image
import path_maker
from tqdm import tqdm
from pathlib import Path
from cnn import CNN
import numpy as np
import matplotlib.pyplot as plt

"""Functions"""


def train(model, dataloader, epoch, len_train_set):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len_train_set/dataloader.batch_size):
        data = data.unsqueeze(1)
        data = data.to(device)
        optimizer.zero_grad()
        recon = model.forward(data)
        loss = model.loss(data, recon)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss


def validate(model, dataloader, epoch, fpath, len_val_set):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len_val_set/dataloader.batch_size):
            data = data.unsqueze(1)
            data = data.to(device)
            recon = model.forward(data)
            loss = model.loss(data, recon)
            running_loss += loss.item()
            if i == 0:
                num_rows = min(data.size(0), 10)
                both = torch.cat(data.view(num_rows, 1, 144, 144)[:10],
                                 recon.view(num_rows, 1, 144, 144)[:10])
                save_image(both.cpu(), "outputs/" +
                           fpath + "/" + str(epoch) + ".png")
    return running_loss


def path_maker(fpath):
    Path("outputs/" + fpath).mkdir(exist_ok=True)


def visualize(fpath, n_samples, train_loss, val_loss):
    figure, axis = plt.subplots(constrained_layout=True)
    axis.plot(np.arange(n_samples), train_loss,
              color="blue", label="Train Loss")
    axis.plot(np.arange(n_samples), val_loss, color="red", label="Val Loss")
    axis.x_label("Number of Training Epochs")
    axis.y_label("Loss")
    figure.legend()
    axis.set_title(fpath + " Loss Visualization.png")


"""Start of training script"""
# Hyperparameters
fpath = "sun_plain"
path_maker(fpath)
epochs = 10
lr = 0.005
batch_size = 32


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

root = ""
data = np.load(root)
data = torch.from_numpy(data).float()


class CustomDataset(Dataset):
    """Planets Dataset"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)


dataset = CustomDataset(data)
train_set, val_set = random_split(dataset, [22500, 2500])

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


t_losses = torch.tensor([0] * epochs)
v_losses = torch.tensor([0] * epochs)
for epoch in range(epochs):
    train_loss = train(model, train_loader, epoch, len_train_set)
    val_loss = validate(model, val_loader, epoch, fpath, len_val_set)
    t_losses[epochs] = train_loss
    v_losses[epochs] = val_loss
    print("Train Loss: " + str(train_loss))
    print("Val Loss: " + str(val_loss))
np.savez(fpath, train_loss=train_loss, val_loss=val_loss)
torch.save(model.state_dict(), "./model_version" + fpath + ".pt")
visualize(fpath, epochs, t_losses, v_losses)
