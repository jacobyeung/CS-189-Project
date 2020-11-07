import torch
import torch.nn as nn
from torchsummary import summary


class VAE(nn.Module):
    """
    Convolutional Beta Variational Autoencoder
    Refer to README for sources and more information about Beta-VAES

    Model encodes 10 latent distributions.
    Model consists of:
        - 3 downsizing convolutional layers
        - 4 linear layers (2 hidden)
        - 3 upsizing convolutional layers
    """

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

        self.enclin_layer = nn.Sequential(
            nn.Linear(800, 400),  # B, 400
            nn.ReLU(),
            nn.Linear(400, 20)  # B, 20
        )
        self.declin_layer = nn.Sequential(
            nn.Linear(10, 400),  # B, 400
            nn.ReLU(),
            nn.Linear(400, 800)  # B, 800
        )
        self.upsize_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=3,
                               padding=1),  # B, 32, 13, 13
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=3,
                               padding=2),  # B, 32, 36, 36
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 6, stride=4,
                               padding=1)  # B, 32, 144, 144
        )

    def encode(self, x):
        x = self.downsize_layer(x)
        x = x.view(-1, 5 * 5 * 32)
        x = self.enclin_layer(x)
        return x

    def decode(self, x):
        x = self.upsize_layer(x)
        x = x.view(-1, 32, 5, 5)
        x = self.upsize_layer(x)
        return x

    def upsize(self, x):
        x = self.upsize_layer(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample

    def forward(self, x):
        x = self.encode(x)
        mu, logvar = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        sample = self.reparameterize(mu, logvar)
        x = self.decode(x)
        return x, mu, logvar

    def final_loss(self, reconstruction, x, mu, logvar, gamma, C):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        bce_loss = criterion(reconstruction, x)
        kld = gamma * \
            torch.abs((-0.5 * torch.mean(1 + logvar -
                                         mu.pow(2) - logvar.exp(), dim=1) - C).mean(dim=0))
        return bce_loss, kld, bce_loss + kld
