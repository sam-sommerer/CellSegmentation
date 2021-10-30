import os
import model

import torch
from torch import nn
from torch import optim

EPOCHS = 100

if __name__ == "__main__":

  mod = model.SimpleModel(5, 5)
  x = torch.randn((3, 1, 520, 704))
  label = torch.zeros((3, 1, 520, 704))
  label[:, :, 20:40, 100:200] = 1.0

  loss_fn = nn.BCEWithLogitsLoss()
  opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)

  for e in range(EPOCHS):
    logits = mod(x)

    loss = loss_fn(logits, label)
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()






