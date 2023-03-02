import logging

import torch
import torch.nn as nn

from dataset import CaravanImageDataLoader
from model import UNET
from modelmanager import ModelManager

logging.basicConfig(level=logging.INFO)

ACCELERATOR = "cuda"
if ACCELERATOR == "cuda":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
elif ACCELERATOR == "mps":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    dev = "cpu"
amp = True if dev == "cuda" else False



def image_segmentation_accuracy(target, prediction):
    pass


def main():
    trn_tf, val_tf = CaravanImageDataLoader.get_default_transforms(
        height=1280 // 4, width=1920 // 4
    )
    data = CaravanImageDataLoader("./data", 8, trn_tf, val_tf)
    model: UNET = UNET().to(device=dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    mm: ModelManager = ModelManager(model, data, loss_fn, optimizer, "WD", dev, amp=amp)
    mm.train()


if __name__ == "__main__":
    main()
