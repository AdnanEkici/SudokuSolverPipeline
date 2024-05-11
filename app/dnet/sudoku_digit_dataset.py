from __future__ import annotations

import os
import sys

import albumentations as A
import cv2
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import utils.utils as utils  # noqa


class SudokuDigitDataset(Dataset):
    def __init__(self, root_dir, logger, transform=None, enable_augmentations: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.augmentations = A.Compose([A.CLAHE(p=0.5), A.GaussianBlur(p=0.4), A.RandomRotate90(p=0.5)])
        self.logger = logger
        self.enable_augmentations = enable_augmentations
        self.logger.debug("Class : Label")
        self.logger.debug(f"Label mapping {self.class_to_idx}")
        self.images = self.make_dataset()

    def make_dataset(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        # Load image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = (
            self.augmentations(image=image)["image"]
            if self.enable_augmentations and (utils.label_mapper(label) != 9 and utils.label_mapper(label) != 6)
            else image
        )
        if self.transform:
            image = self.transform(image)
        return image, label
