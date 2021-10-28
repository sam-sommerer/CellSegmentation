import cv2
import math
import numpy as np
import pandas as pd

colors_dict = {
    "shsy5y": np.array([255, 0, 0]),
    "astro": np.array([0, 255, 0]),
    "cort": np.array([0, 0, 255])
}

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

first_id = ids[0]

final = np.zeros((520, 704, 3))

for i in range(len(ids)):
  print(i)
  if ids[i] != first_id:
    break
  decoded = decodeRLE(annotations[i], types[i], (520, 704, 3))
  final += decoded

mask = (final > 0).astype(np.uint8) * 255

cv2.imshow("test", mask)
cv2.waitKey(0)

cv2.destroyAllWindows()









