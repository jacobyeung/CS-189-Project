import torch
import unittest
from reimp import ReImp
from torchsummary import summary


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = ReImp()

    def test_summary(self):
        print(summary(self.model, (1, 64, 64), device='cpu'))

    def test_encoder(self):
        x = torch.randn(64, 1, 64, 64)
        y = self.model.encode(x)
        print("Encoder output size: ", y.size())

    def test_decoder(self):
        x = torch.randn(10)
        y = self.model.decode(x)
        print("Decoder output size: ", y[0].size())

    def test_forward(self):
        x = torch.randn(64, 1, 64, 64)
        z = x.detach().clone()
        for layer in self.model.enc_convLayer:
            z = layer(z)
            print(z.size())
        z = z.view(-1, 4 * 4 * 32)
        for layer in self.model.enc_linLayer:
            z = layer(z)
            print(z.size())

        y = self.model(x)
        print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(64, 1, 64, 64)

        recon, mu, logvar = self.model(x)
        loss = self.model.final_loss(recon, x, mu, logvar, 1000, 25)
        print(str(loss) + 'hi')


if __name__ == '__main__':
    unittest.main()
