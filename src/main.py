import torch
import torch.nn as nn
from dataset import CaravanImageDataLoader
from model import ModelManager
from model import UNET

import logging
logging.basicConfig(level=logging.INFO)

ACCELERATOR = "mps"
if ACCELERATOR == "cuda":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
elif ACCELERATOR == "mps":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    dev = "cpu"

amp = True if dev == "cuda" else False


def train_model():
    # dataset = CaravanImageDataset("./caravan_images")
    # model = UNET()
    pass


def image_segmentation_accuracy(target, prediction):
    pass


def main():
    trn_tf, val_tf = CaravanImageDataLoader.get_default_transforms(
        height=1280 // 4, width=1920 // 4
    )
    data = CaravanImageDataLoader("./data", 8, trn_tf, val_tf)
    model: UNET = UNET().to(device=dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mm: ModelManager = ModelManager(model, data, loss_fn, optimizer, "WD", dev, amp=amp)
    for i in range(5):
        trn_loss, val_loss = mm.train_epoch()
        print(f"[{i}] trn_loss {trn_loss}, val_loss {val_loss}")
    # Size: batch_size, nFeatures, Heigh, Width
    # tt = torch.rand((2, 1, 572, 572))
    # model.forward(tt)


if __name__ == "__main__":
    main()
