import torch.nn as nn
import torch.nn.functional as F
import torch


class DCUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DCUnit, self).__init__()
        self.dcunit = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.dcunit(x)


class CUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CUnit, self).__init__()
        self.cunit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.cunit(x)


class Generator(nn.Module):
    def __init__(self, latent_size, n_features):
        super(Generator, self).__init__()
        self.dcnn = nn.Sequential(
            DCUnit(latent_size, 4 * n_features, 4, 1, 0),
            DCUnit(4 * n_features, 2 * n_features, 4, 2, 1),
            DCUnit(2 * n_features, n_features, 4, 2, 1),

            nn.ConvTranspose2d(n_features, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            nn.Upsample(size=(18, 11))
        )

    def forward(self, x):
        return self.dcnn(x)


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Upsample(size=(32, 32)),

            CUnit(3, n_features, 4, 2, 1),
            CUnit(n_features, 2 * n_features, 4, 2, 1),
            CUnit(2 * n_features, 4 * n_features, 4, 2, 1),

            nn.Conv2d(4 * n_features, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)
