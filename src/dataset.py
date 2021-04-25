from pathlib import Path 
from glob import glob
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.histograms import _histogram_bin_edges_dispatcher

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import cv2

class CaravanImageDataset(Dataset):
    def __init__(self, image_path: Path, mask_path: Path, transform: Compose = None):
        self._image_path = image_path.absolute()
        self._mask_path = mask_path.absolute()
        self._transform = transform

        self.images = sorted(glob(str(self._image_path.joinpath("*.jpg"))))
        self.nImages = len(self.images)
        self.masks = sorted(glob(str(self._mask_path.joinpath("*_mask.jpg"))))

        assert self.nImages > 0, "Could not find images!"
        assert len(self.masks) == self.nImages, "Number of masks and images must be the same!"
    
    def __len__(self):
        return self.nImages
    
    def __getitem__(self, index : int) -> Tuple[List, List]:
        imgpath = self.images[index%self.nImages]
        maskpath = self.masks[index%self.nImages]

        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(maskpath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Convert mask [0, 255] to [0, 1]
        mask = mask / 255.0

        if self._transform:
            augmented = self._transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def sample(self, N=3):
        fig, axs = plt.subplots(N, 2, figsize=(10, 10))
        for (ax1, ax2) in axs:
            ax1.set_axis_off()
            ax2.set_axis_off()
            n = int(np.random.random()*self.nImages)
            image, mask = self[n]
            ax1.imshow(image)
            ax2.imshow(mask)
        return fig

@dataclass
class CaravanImageDataLoader():
    TRAIN = "train"
    TRAIN_MASKS = "train_masks"
    VAL = "validation"
    VAL_MASKS = "validation_masks"

    training_loader: DataLoader
    validation_loader: DataLoader

    def __init__(self, dataset_base_path:str, batch_size: int,
        train_transform: Compose = None,
        valid_transform: Compose = None
        ):
       self._training_dataset = CaravanImageDataset(Path(dataset_base_path).joinpath(self.TRAIN), Path(dataset_base_path).joinpath(self.TRAIN_MASKS), train_transform) 
       self.training_loader = DataLoader(self._training_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

       self._validation_dataset = CaravanImageDataset(Path(dataset_base_path).joinpath(self.VAL), Path(dataset_base_path).joinpath(self.VAL_MASKS), valid_transform) 
       self.validation_loader = DataLoader(self._validation_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,)

    @staticmethod 
    def get_default_transforms(height: int = 1280//4, width: int = 1918//4) -> Tuple[Compose, Compose]:
        train_transform = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
        )

        val_transforms = A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        return train_transform, val_transforms