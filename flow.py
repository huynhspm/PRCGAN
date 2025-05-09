from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import math

class CondFlow(nn.Module):
    def __init__(self, img_dims: Tuple[int, int, int]):
        super().__init__()
        inp_dim = math.prod(img_dims) * 2 + 1  # x_t + condition + time t
        self.img_dims = img_dims
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.net = nn.Sequential(
            *block(inp_dim, 2048, normalize=False),
            *block(2048, 2048),
            *block(2048, 2048),
            *block(2048, 2048),
            nn.Linear(2048, math.prod(img_dims))
        )
    
    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        # x_t: [B, dim], t: [B,1], cond: [B, cond_dim]
        B, _, _, _ = x_t.shape
        inp = torch.cat([x_t.view(B, -1), cond.view(B, -1), t], dim=1)
        return self.net(inp).view(B, *self.img_dims)
    
    def step(self, x_t: Tensor, t0: Tensor, t1: Tensor, cond: Tensor) -> Tensor:
        # Midpoint solver
        t0 = t0.view(1, 1).expand(x_t.shape[0], 1)
        t1 = t1.view(1, 1).expand(x_t.shape[0], 1)
        dt = (t1 - t0)
        k1 = self(x_t, t0, cond)
        mid = x_t + k1 * (dt * 0.5).view(-1, 1, 1, 1)
        k2 = self(mid, t0 + dt*0.5, cond)
        return x_t + k2 * dt.view(-1, 1, 1, 1)

if __name__ == "__main__":
    img_dims = [1, 28, 28]

    flow = CondFlow(img_dims)
    img = torch.randn(10, 1, 28, 28)
    t = torch.rand(len(img), 1)
    out = flow(img, t, cond = img)
    print(out.shape)