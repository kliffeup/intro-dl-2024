import os
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm


def extract_cars_from_bboxes(data_path: str, bboxes_path: str) -> dict[str, np.ndarray]:
    bboxes = pd.read_csv(bboxes_path, index_col="filename")
    filename2image = dict()

    for dirname, _, filenames in os.walk(data_path):
        for filename in tqdm.tqdm(filenames):
            filename2image[filename] = cv2.cvtColor(cv2.imread(os.path.join(dirname, filename)), cv2.COLOR_BGR2RGB)
            ymin = max(int(bboxes.loc[filename].ymin), 0)
            ymax = int(bboxes.loc[filename].ymax)
            xmin = max(int(bboxes.loc[filename].xmin), 0)
            xmax = int(bboxes.loc[filename].xmax)
            filename2image[filename] = filename2image[filename][ymin:ymax, xmin:xmax]

    return filename2image


def apply_augmentations(image: np.ndarray, augmentation_pipeline) -> np.ndarray:
    augmented = augmentation_pipeline(image=image)
    return augmented["image"]


def calc_norm_constants(images: Iterable[np.ndarray], augmentation_pipeline, num_augs_per_image: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per channel statistics."""
    k = 0
    sum_img = 0
    sum_sq_img = 0

    for img in tqdm(images):
        for _ in range(num_augs_per_image):
            augmented_image = apply_augmentations(img, augmentation_pipeline) / 255
            k += 1
            sum_img += augmented_image
            sum_sq_img += augmented_image ** 2

    mean = np.mean(sum_img / k, axis=(0, 1))
    std = np.mean((sum_sq_img - sum_img ** 2 / k) / (k - 1), axis=(0, 1))
    return mean, std


class CarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, filename2image: dict[str, np.ndarray], train_transforms=None, val_transforms=None) -> None:
        self.df = df
        self.filename2image = filename2image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        filename = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['label']

        image = self.filename2image[filename]

        if self.train_transforms is not None:
            augmented = self.train_transforms(image=image)
            image = augmented['image']

        if self.val_transforms is not None:
            augmented = self.val_transforms(image=image)
            image = augmented['image']
        
        image /= 255

        return image, label
