import os
import cv2
import math
import numpy as np
import pandas as pd

colors_dict = {
    "shsy5y": np.array([255, 0, 0]),
    "astro": np.array([0, 255, 0]),
    "cort": np.array([0, 0, 255])
}

DATA_DIR = "data"

def decodeRLE(encoded, cell_type, im_shape):
  nums = [int(num) for num in encoded.split(" ")]
  chunk = np.zeros(im_shape).reshape(-1, im_shape[-1]).astype(np.uint8)
  i = 0
  while i < len(nums) - 1:
    pos_init = nums[i]
    n = nums[i+1]
    i += 2
    chunk[pos_init:pos_init + n, :] = colors_dict[cell_type]
  return chunk.reshape(im_shape)   

data = pd.read_csv("data/train.csv")

ids = data["id"]
types = data["cell_type"]
annotations = data["annotation"]
width = data["width"]
height = data["height"]

cur_id = ids[0]
cur_img = 0
Xs = np.zeros((ids.nunique(), 520, 704))
ys = np.zeros((ids.nunique(), 520, 704))
final = np.zeros((520, 704, 3))

for i in range(len(ids)):
  if ids[i] != cur_id:
    '''
    # Visualization:
    inp = cv2.imread(os.path.join(DATA_DIR, "train", f"{ids[i-1]}.png"))
    mask = (final > 0).astype(np.uint8) * 255
    comb = (inp + mask / 2).astype(np.uint8)

    cv2.imshow("test", inp)
    key = cv2.waitKey(0)
    cv2.imshow("test", mask)
    key = cv2.waitKey(0)
    cv2.imshow("test", comb)
    key = cv2.waitKey(0)

    if key == ord("q"):
      cv2.destroyAllWindows()
      quit()
    '''
    print(cur_img)

    Xs[cur_img] = cv2.imread(os.path.join(DATA_DIR, "train", f"{ids[i-1]}.png"))[:, :, 0]
    ys[cur_img] = (np.sum(final, axis=2) > 0).astype(np.float32)

    # Reset
    cur_id = ids[i]
    cur_img += 1
    final = np.zeros((520, 704, 3))

  decoded = decodeRLE(annotations[i], types[i], (520, 704, 3))
  final += decoded

# Xs -= Xs.mean()
# Xs /= Xs.std()

np.save("train_X.npy", Xs)
np.save("train_y.npy", ys)


