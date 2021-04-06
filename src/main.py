from model import UNET
from dataset import CaravanImageDataset

def train_model():
    dataset = CaravanImageDataset("./caravan_images")
    model = UNET()

if __name__ == "__main__":
    pass