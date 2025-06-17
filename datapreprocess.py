import os
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import albumentations as A

H, W = 256, 256

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.RandomBrightnessContrast(p=0.2),
], additional_targets={'mask': 'mask'})

def load_dataset(dataset_path, val_ratio=0.2, test_ratio=0.1):
    images = sorted(glob(os.path.join(dataset_path, 'images', '*.png')))
    masks = sorted(glob(os.path.join(dataset_path, 'masks', '*.png')))
    assert len(images) == len(masks), "Numbers of images and masks does not match"

    X_temp, X_test, y_temp, y_test = train_test_split(images, masks, test_size=test_ratio, random_state=42)

    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_image(path, target_size=(H, W)):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, target_size)
    x = x/255.0
    
    return x.astype(np.float32)

def load_mask(path, target_size=(H, W)):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, target_size)
    x = x/255.0

    return np.expand_dims(x.astype(np.float32), axis=-1)

def augment(image, mask):
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']

def data_generator(X, Y, batch_size = 32, augment_fn=None):
    while True:
        idxs = np.random.choice(len(X), batch_size)
        batch_x = []
        batch_y = []
        for i in idxs:
            img = load_image(X[i])
            msk = load_mask(Y[i])
            if augment_fn is not None:
                img, msk = augment_fn(img, msk)
            batch_x.append(img)
            batch_y.append(msk)
        yield np.array(batch_x), np.array(batch_y)

if __name__ == '__main__':
    data_path = "E:/brain_tumor_segmentation/dataset"
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_dataset(data_path)

    print(f"Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")

    img = load_image(train_x[0])
    mask = load_mask(train_y[0])
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")