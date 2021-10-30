import os
import numpy as np

import torch
from torch import nn
from torch import functional as F

def conv2drelu(f_maps):
  return nn.Sequential(nn.Conv2d(f_maps, f_maps, 3, padding=1), nn.ReLU())

class SimpleModel(nn.Module):
  def __init__(self, n_convs, f_maps):
    super().__init__()

    self.init_conv = nn.Conv2d(1, f_maps, 3, padding=1)
    self.convs = nn.Sequential(*[conv2drelu(f_maps) for _ in range(n_convs)])
    self.final_conv = nn.Conv2d(f_maps, 1, 3, padding=1)
    self.softmax2d = nn.Softmax2d()

  def forward(self, x):
    x = self.init_conv(x).relu()
    x = self.convs(x)
    x = self.final_conv(x)
    x = self.softmax2d(x)
    return x


if __name__ == "__main__":
  x = torch.randn((1, 1, 520, 704))
  mod = SimpleModel(5, 10)
  pred = mod(x)
  print(pred)
