from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import numpy
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from numpy.lib.histograms import _histogram_bin_edges_dispatcher

def create_figure_of_image_mask_pairs(pairs: List[Tuple[numpy.ndarray, numpy.ndarray]]) -> Figure:
    fig, axs = plt.subplots(len(pairs), 2, figsize=(10, 10))
    for (image, mask), (ax1, ax2) in zip(pairs, axs):
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1.imshow(image)
        ax2.imshow(mask, cmap='gray', vmin=0, vmax=255)
    return fig


class CaravanImageDataset(Dataset):
    # Optimized for the kaggle dataset: https://www.kaggle.com/competitions/carvana-image-masking-challenge/data
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        imgpath = self.images[index % self.nImages]
        maskpath = self.masks[index % self.nImages]

        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(maskpath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Convert mask [0, 255] to [0, 1]
        # Image will be normalized in transformation.
        # The Albumentation api describes `targets` for each
        # transformation and the normalization transformation only has **image**
        # as a target, not **mask**
        mask = mask / 255.0
        #print(f"Mask min-max {mask.min()}-{mask.max()}")

        if self._transform:
            augmented = self._transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def sample(self, N=3) -> Figure:
        fig, axs = plt.subplots(N, 2, figsize=(10, 10))
        for ax1, ax2 in axs:
            ax1.set_axis_off()
            ax2.set_axis_off()
            n = int(np.random.random() * self.nImages)
            image, mask = self[n]
            ax1.imshow(image)
            ax2.imshow(mask)
        return fig


@dataclass
class CaravanImageDataLoader:
    TRAIN = "train"
    TRAIN_MASKS = "train_masks"
    VAL = "validation"
    VAL_MASKS = "validation_masks"

    training_loader: DataLoader
    validation_loader: DataLoader

    def __init__(self, dataset_base_path: str, batch_size: int, train_transform: Compose, valid_transform: Compose):
        self._training_dataset = CaravanImageDataset(
            Path(dataset_base_path).joinpath(self.TRAIN),
            Path(dataset_base_path).joinpath(self.TRAIN_MASKS),
            train_transform,
        )
        self.training_loader = DataLoader(
            self._training_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True
        )

        self._validation_dataset = CaravanImageDataset(
            Path(dataset_base_path).joinpath(self.VAL),
            Path(dataset_base_path).joinpath(self.VAL_MASKS),
            valid_transform,
        )
        self.validation_loader = DataLoader(
            self._validation_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def get_default_transforms(height: int = 1280 // 4, width: int = 1918 // 4) -> Tuple[Compose, Compose]:
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

class CaravanImage(Dataset):
    """Helper class to load image data for inference
    """
    # Optimized for the kaggle dataset: https://www.kaggle.com/competitions/carvana-image-masking-challenge/data
    def __init__(self, dataset_path: Path, transformation: Compose = None, batch_size: int = 32):
        """Create a pytorch dataset from the given path containing (training, training_masks, validation, validation_masks)
        Apply given transformations when loading the images during training.
        For image augmentation, see [here](https://albumentations.ai/docs/examples/example/#Define-an-augmentation-pipeline-using-Compose,-pass-the-image-to-it-and-receive-the-augmented-image)

        Args:
            dataset_path (Path): file path to training data folder (containing separate subfolders)
            transformation (Compose, optional): albumentations image augmentations. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 32.
        """
        self._image_path = dataset_path.absolute()

        if transformation is None:
            # Get the validation transformation
            _, transformation = CaravanImageDataLoader.get_default_transforms()
        self.transformation = transformation

        self.images = sorted(glob(str(self._image_path.joinpath("*.jpg"))))
        self.nImages = len(self.images)

        assert self.nImages > 0, "Could not find images!"

        self.data_loader = DataLoader(
            self, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False
        )

    def __len__(self):
        return self.nImages

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        imgpath = self.images[index % self.nImages]

        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transformation(image=image)["image"]

        return tensor 

