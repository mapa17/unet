import torch

from model import UNET
#from dataset import CaravanImageDataset

def train_model():
    #dataset = CaravanImageDataset("./caravan_images")
    #model = UNET()
    pass

if __name__ == "__main__":
    model = UNET()
    # Size: batch_size, nFeatures, Heigh, Width
    tt = torch.rand((2, 1, 572, 572))
    model.forward(tt)

    pass