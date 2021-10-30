import os
import math
import numpy as np

import torch
from torch import nn
from torch import optim

from model import SimpleModel

EPOCHS = 100
BATCH_SIZE = 5
DATA_DIR = "/storage/cell_seg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

  mod = SimpleModel(10, 10)

  Xs = torch.tensor(np.load(os.path.join(DATA_DIR, "train_X.npy")).reshape((606, 1, 520, 704)).astype(np.float32))
  ys = torch.tensor(np.load(os.path.join(DATA_DIR, "train_y.npy")).reshape((606, 1, 520, 704)).astype(np.float32))

  loss_fn = nn.BCEWithLogitsLoss()
  opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)

  batch_count = math.ceil(len(Xs) / BATCH_SIZE)

  for e in range(EPOCHS):
    print(f"Epoch {e}...")
    for batch in range(batch_count):
      batch_X = Xs[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE].to(DEVICE)
      batch_y = ys[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE].to(DEVICE)
      logits = mod(batch_X)
      loss = loss_fn(logits, batch_y)
      print(loss)
      opt.zero_grad()
      loss.backward()
      opt.step()
