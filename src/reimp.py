import torch
import torch.nn as nn
from torchsummary import summary
# torch.manual_seed(4321234)


class ReImp(nn.Module):
    """Reimplmentation of paper"""

    def __init__(self):
        super(ReImp, self).__init__()

        # Encoder

        self.enc_convLayer = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # B, 32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 4, 4
            nn.ReLU()
        )

        self.enc_linLayer = nn.Sequential(
            nn.Linear(4 * 4 * 32, 256),                 # B, 256
            nn.ReLU(),
            nn.Linear(256, 256),                        # B, 256
            nn.ReLU(),
            nn.Linear(256, 20)                          # B, 20
        )

        # Decoder
        self.dec_linLayer = nn.Sequential(
            nn.Linear(10, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, 4 * 4 * 32),  # B, 512
            nn.ReLU()
        )

        self.dec_convLayer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # B, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2,
                               padding=1),                      # B, 32, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2,
                               padding=1),                      # B, 32, 32, 32

            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)  # B, 1, 64, 64
        )

    def encode(self, x):
        x = self.enc_convLayer(x)                       # Encode to B, 32, 4, 4
        x = x.view(-1, 4 * 4 * 32)                      # B, 512
        x = self.enc_linLayer(x)
        return x

    def decode(self, sample):
        x = self.dec_linLayer(sample)  # B, 512
        x = x.view(-1, 32, 4, 4)  # B, 32, 4, 4
        x = self.dec_convLayer(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample

    def forward(self, x):
        x_size = x.size()
        x = self.encode(x)
        print(x.shape)
        mu, logvar = x[:, :10], x[:, 10:]  # split latent layer in half
        sample = self.reparameterize(mu, logvar)
        print(sample.shape)
        x = self.decode(sample)
        x = x.view(x_size)
        return x, mu, logvar

    def final_loss(self, reconstruction, x, mu, logvar, gamma, C):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        bce_loss = criterion(reconstruction, x)
        kld = gamma * \
            torch.abs((-0.5 * torch.mean(1 + logvar -
                                         mu.pow(2) - logvar.exp(), dim=1) - C).mean(dim=0))
        return bce_loss, kld, bce_loss + kld
