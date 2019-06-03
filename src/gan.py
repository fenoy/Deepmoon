import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_size, n_features):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3*18*11),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.squeeze()
        img = self.model(z)
        img = img.view(img.size(0), 3, 18, 11)
        return img

class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3*18*11, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


