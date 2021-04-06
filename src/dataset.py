from torch.utils.data import Dataset
import cv2
import albumentations as A
from matplotlib import pyplot as plt
from pathlib import Path 
import numpy as np
from glob import glob

class CaravanImageDataset(Dataset):
    TRAIN = "train"
    TRAIN_MASKS = "train_masks"
    VAL = "validatation"
    VAL_MASKS = "validation_masks"

    def __init__(self, dataset_base_path:str):
        self.train_path = Path(dataset_base_path).absolute().joinpath(TRAIN)
        self.train_masks_path = Path(dataset_base_path).absolute().joinpath(TRAIN_MASKS)

        self.val_path = Path(dataset_base_path).absolute().joinpath(VAL)
        self.val_masks_path = Path(dataset_base_path).absolute().joinpath(VAL_MASKS)

        self.images = glob(str(self.train_path.joinpath("*.jpg")))
        self.masks = glob(str(self.train_masks_path.joinpath("*_mask.gif")))
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index : int) -> tuple(List, List):
        pass

    def sample(self, N=9):
        pass
"""
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
visualize(image)
plt.show()
visualize(image2)
"""
