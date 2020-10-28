import torch
import torch.nn as nn
from torchsummary import summary


class CRNN(nn.Module):
    """Convolutional recurrent neural network using LSTM"""

    def __init__(self):
        super().__init__()

        # Downsize Conv
        self.downsize_convLayer = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # B, 32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 8, 8
            nn.ReLU()
        )

        self.reccurent_layer = nn.LSTM(32, 32, num_layers=2)

    def downsize(self, x):
        x = self.downsize_convLayer(x)
        return x

    def recurrent(self, x):
        x = self.reccurent_layer(x)
        return x

    def forward(self, x):
        print(x.shape)
        x = self.downsize(x)
        print(x.shape)
        x = self.recurrent(x)
        return x
