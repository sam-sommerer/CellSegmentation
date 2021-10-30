import albumentations as A
import numpy as np
import cv2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

imgs_file = "data/train_X.npy"
masks_file = "data/train_y.npy"
aug_imgs_file = "data/aug_train_X.npy"
aug_masks_file = "data/aug_train_Y.npy"

def augment(imgs_file, masks_file, transform):
    imgs = np.load(imgs_file)
    masks = np.load(masks_file)

    imgs = imgs.astype(np.uint8)
    masks = masks.astype(np.uint8)

    aug_imgs = []
    aug_masks = []

    for img, mask in zip(imgs, masks):
        transformed = transform(image=img, mask=mask)
        aug_imgs.append(transformed["image"])
        aug_masks.append(transformed["mask"])
    
    return aug_imgs, aug_masks

aug_imgs, aug_masks = augment(imgs_file, masks_file, transform)

np.save(aug_imgs_file, np.asarray(aug_imgs))
np.save(aug_masks_file, np.asarray(aug_masks))

