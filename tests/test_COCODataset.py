from pathlib import Path

from COCODataset import COCODataset


def test_COCODataset():
    #trn_tf, _ = COCODataset.get_default_transforms(height=1280 // 4, width=1920 // 4)
    ds = COCODataset(Path("data"))
    image, mask = ds[0]
    assert image is not None and mask is not None, "Could not load images ..."
