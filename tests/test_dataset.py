from pathlib import Path

from src.dataset import CaravanImageDataLoader
from src.dataset import CaravanImageDataset


def test_CaravanImageDataset():
    trn_tf, _ = CaravanImageDataLoader.get_default_transforms(height=1280 // 4, width=1920 // 4)
    ds = CaravanImageDataset(Path("data/train"), Path("data/train_masks"), trn_tf)
    image, mask = ds[0]
    assert image is not None and mask is not None, "Could not load images ..."
