import os
import numpy as np

import torch
from torch import nn
from torch import optim

from model import SimpleModel

EPOCHS = 100
DATA_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

  mod = SimpleModel(10, 10)

  Xs = torch.tensor(np.load(os.path.join(DATA_DIR, "train_X.npy")).reshape((606, 1, 520, 704)).astype(np.float32))
  ys = torch.tensor(np.load(os.path.join(DATA_DIR, "train_y.npy")).reshape((606, 1, 520, 704)).astype(np.float32))
  Xs.to(DEVICE)
  ys.to(DEVICE)

  loss_fn = nn.BCEWithLogitsLoss()
  opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)

  for e in range(EPOCHS):
    logits = mod(Xs)

    loss = loss_fn(logits, ys)
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
