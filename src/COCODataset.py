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

from pycocotools.coco import COCO

# from numpy.lib.histograms import _histogram_bin_edges_dispatcher

def create_figure_of_image_mask_pairs(pairs: List[Tuple[numpy.ndarray, numpy.ndarray]]) -> Figure:
    fig, axs = plt.subplots(len(pairs), 2, figsize=(10, 10))
    for (image, mask), (ax1, ax2) in zip(pairs, axs):
        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1.imshow(image)
        ax2.imshow(mask, cmap='gray', vmin=0, vmax=255)
    return fig


from typing import NamedTuple

class COCOImage(NamedTuple):
    id: int
    filename: str
    annotations: List[dict]


class COCODataset(Dataset):
    """COCO Dataset
    www.cocodataset.org

    An image segmentation dataset with 80 object categories.
    This data loader is making use of the pycocotools library (https://github.com/ppwwyyxx/cocoapi)

    To load image data, annotations and meta data

    The dataset contains jpg images and json annotations.
    - 118287 training images
    - 5000 validation images
    - 40670 test images (no annotations exist for the test images)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, annotation_filepath: Path, image_filepath: Path, transform: Compose = None):
        self._image_path = image_filepath.absolute()
        self._images = COCODataset.__load_images(annotation_filepath.absolute())
        self._transform = transform
        self.nImages = len(self._images)
        self.coco = COCO()
        
        assert self.nImages > 0, "Could not find images!"
        

    @staticmethod
    def __load_images(annotation_file_path: Path) -> list[COCOImage]:
        coco = COCO(annotation_file_path)
        imgs: List[COCOImage] = []

        for img_id, meta in coco.imgs.items():
            # A single image can have multiple segmentation.
            # For each segmentation there is a separate annotation object.
            # In a single image you can multiple segmentation of the same category
            annIds = coco.getAnnIds(img_id)
            annotations = coco.loadAnns(annIds)
            imgs.append(COCOImage(id=img_id, filename=meta['file_name'], annotations=annotations))
        return imgs

    def __create_mask(self, annotation):
        # TODO: Create a mask for each channel based on the annotation object
        pass

    def __len__(self):
        return self.nImages

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = index % self.nImages
        img = self._images[index]
        mask = self.__create_mask(img.annotations)
        
        image = cv2.imread(self._image_path / img.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
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
class COCODataLoader:
    TRAIN = "train"
    TRAIN_MASKS = "train_masks"
    VAL = "validation"
    VAL_MASKS = "validation_masks"

    training_loader: DataLoader
    validation_loader: DataLoader

    def __init__(self, dataset_base_path: str, batch_size: int, train_transform: Compose, valid_transform: Compose):
        trn_transform, val_transform = self.get_default_transforms()

        self._training_dataset = COCODataset(
            Path(dataset_base_path).joinpath(self.TRAIN),
            Path(dataset_base_path).joinpath(self.TRAIN_MASKS),
            train_transform or trn_transform
        )
        self.training_loader = DataLoader(
            self._training_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True
        )

        self._validation_dataset = COCODataset(
            Path(dataset_base_path).joinpath(self.VAL),
            Path(dataset_base_path).joinpath(self.VAL_MASKS),
            valid_transform or val_transform
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

