import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    """Convolutional recurrent neural network using LSTM"""

    def __init__(self):
        super().__init__()

        # Downsize Conv Image size 144x144
        self.downsize_layer = nn.Sequential(
            nn.Conv2d(1, 32, 6, stride=4, padding=1),   # B, 32, 36, 36
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=3, padding=2),  # B, 32, 13, 13
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=3, padding=1),  # B, 32, 5, 5
            nn.ReLU()
        )

        self.lin_layer = nn.Sequential(
            nn.Linear(800, 800)  # B, 32, 5, 320
        )

        self.upsize_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 6, stride=4, padding=1),
        )

    def downsize(self, x):
        x = self.downsize_layer(x)
        return x

    def linear(self, x):
        x = self.lin_layer(x)
        return x

    def upsize(self, x):
        x = self.upsize_layer(x)
        return x

    def forward(self, x):
        x = self.downsize(x)
        x = x.view(-1, 5 * 5 * 32)
        x = self.linear(x)
        x = x.view(-1, 32, 5, 5)
        x = self.upsize(x)
        return x

    def loss(self, x, reconstruction):
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        bce_loss = criterion(reconstruction, x)
        return bce_loss
