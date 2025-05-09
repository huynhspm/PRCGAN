from typing import Tuple, Dict

import math
import torch
import torch.nn as nn
from torch import Tensor

class Generator(nn.Module):
    def __init__(self, 
                img_dim: int, 
                d_cond: int | None = None) -> None:
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_dim + (d_cond if d_cond is not None else 0), 2048, normalize=False),
            *block(2048, 2048),
            *block(2048, 2048),
            *block(2048, 2048),
            nn.Linear(2048, img_dim),
            nn.Tanh()
        )

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        if cond is not None:
            z = torch.cat((z, cond), dim=1)
        
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, 
                img_dim, 
                d_cond: int | None = None) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(img_dim + (d_cond if d_cond is not None else 0), 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, img: Tensor, cond: Tensor) -> Tensor:
        if cond is not None:
            img = torch.cat((img, cond), dim=1)

        validity = self.model(img)
        return validity

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class CGAN(nn.Module):
    def __init__(self,
                img_dims: Tuple[int, int, int],
                cond_dims: Tuple[int, int, int]) -> None:
        super().__init__()

        self.gen = Generator(img_dim=math.prod(img_dims),
                            d_cond=math.prod(cond_dims))
        self.gen.apply(weights_init)

        self.disc = Discriminator(img_dim=math.prod(img_dims),
                                d_cond=math.prod(cond_dims))
        self.disc.apply(weights_init)
        self.img_dims = img_dims

    def classify(self, cond: Tensor, image: Tensor) -> Tensor:
        return self.disc(image.view(image.size(0), -1), 
                        cond.view(cond.size(0), -1))

    def sample(self,
            num_sample: int,
            cond: Tensor,
            device: torch.device = torch.device("cpu")) -> Tensor:

        z = torch.randn(num_sample, *self.img_dims, device=device)
        image = self.gen(z.view(z.size(0), -1), 
                        cond.view(cond.size(0), -1))

        return image.view(num_sample, *self.img_dims)

if __name__ == "__main__":
    img_dims = [1, 28, 28]

    gan = CGAN(img_dims=img_dims, cond_dims=[1, 28, 28])

    cond = torch.randn(10, 1, 28, 28)
    image = gan.sample(num_sample=10, cond=cond)
    print(image.shape)

    label = gan.classify(image=image, cond=cond)
    print(label.shape)

