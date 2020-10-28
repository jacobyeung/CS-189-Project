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
                num_rows = min(data.size(0), 9)
                both = torch.cat((data.view(num_rows, 1, 144, 144)[:5],
                                  recon.view(num_rows, 1, 144, 144)[:5]))
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


"""Start of training script"""
# Hyperparameters
fpath = "Jupiter"
path_maker(fpath)
epochs = 40
lr = 0.005
batch_size = 32


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

root = "combined_data_matrix/Jupiter.npz"
data = np.load(root)
data = 1 * data['data']

data = torch.from_numpy(data).float()
total_len = len(data)


class CustomDataset(Dataset):
    """Planets Dataset"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)


dataset = CustomDataset(data)
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


t_losses = torch.tensor([0] * epochs)
v_losses = torch.tensor([0] * epochs)
for epoch in range(epochs):
    train_loss = train(model, train_loader, epoch, len_train_set)
    val_loss = validate(model, val_loader, epoch, fpath, len_val_set)
    t_losses[epoch] = train_loss
    v_losses[epoch] = val_loss
    print("Train Loss: " + str(train_loss))
    print("Val Loss: " + str(val_loss))
np.savez(fpath, train_loss=train_loss, val_loss=val_loss)
torch.save(model.state_dict(), "./model_version" + fpath + ".pt")
visualize(fpath, epochs, t_losses, v_losses)
