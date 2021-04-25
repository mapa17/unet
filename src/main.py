import torch
import torch.nn as nn
from model import ModelManager, UNET
from dataset import CaravanImageDataLoader, CaravanImageDataset

def train_model():
    #dataset = CaravanImageDataset("./caravan_images")
    #model = UNET()
    pass

def image_segmentation_accuracy(target, prediction):
    pass

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pudb import set_trace as st; st()
    trn_tf, val_tf = CaravanImageDataLoader.get_default_transforms(height=1280//4, width=1920//4)
    data = CaravanImageDataLoader("../caravan_images", 2, trn_tf, val_tf)
    model = UNET().to(device=dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mm = ModelManager(model, data, loss_fn, optimizer, 'WD')
    mm.train_epoch()
    # Size: batch_size, nFeatures, Heigh, Width
    #tt = torch.rand((2, 1, 572, 572))
    #model.forward(tt)

    pass