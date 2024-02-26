import logging
import sys
from typing import Callable, Tuple
from pathlib import Path
import os

import torch
import torch.nn as nn
import click
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from dataset import CaravanImageDataLoader, CaravanImage, create_figure_of_image_mask_pairs
from model import UNET
from trainingmanager import TrainingManager


__version__ = "0.1"
log = None

def select_device(preference: str) -> Tuple[torch.device, bool]:
    if preference == "cuda":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif preference == "mps":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = "cpu"
    amp = True if dev == "cuda" else False
    return dev, amp



def image_segmentation_accuracy(target, prediction):
    # TODO: Implement 1-2 image segmentation accuracy functions
    pass


def load_model(
    model_path: str, model_constructor: Callable[[], nn.Module], dev:torch.device
) -> nn.Module:
    """Generic pytorch model loader

    Args:
        model_path (str): Path to checkpoint
        model_constructor (Callable[[], nn.Model]): Model Class

    Returns:
        nn.Model: _description_
    """
    model: nn.Model = model_constructor().to(device=dev)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _train(data_folder: str, max_epochs: int, workdir: str, dev: torch.device, amp: bool):
    """Load a CaravanImage Dataset and Train a UNET with the TrainingManager

    Args:
        data_folder (str): Training Dataset folder 
        max_epochs (int): train up to `max_epochs` epochs
        workdir (str): path to working directory storing logs and checkpoints
        dev (torch.device): torch device to use for training
        amp (bool): if True, use automatic mixed precision training
    """
    trn_tf, val_tf = CaravanImageDataLoader.get_default_transforms(
        height=1280 // 4, width=1920 // 4
    )
    data = CaravanImageDataLoader(data_folder, 8, trn_tf, val_tf)
    model: UNET = UNET().to(device=dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    tm: TrainingManager = TrainingManager(
        model, data, loss_fn, optimizer, workdir, dev=dev, amp=amp
    )
    tm.train(max_epoch=max_epochs)


def _run(model_path: str, data_folder: str, dev: torch.device) -> None:
    model = load_model(model_path, UNET, dev)
    model.eval()
    data = CaravanImage(Path(data_folder)).data_loader
    log.info(f"Found {len(data)} images!")
    masks = []
    with torch.no_grad():
        with tqdm(data, desc="Inference") as images:
            for i, image in enumerate(images):
                _image = image.to(device=dev)

                pred_mask = model(_image)

                # squeeze(1) ... get rid of the second axis if it is of dimension 1
                minibatch_masks = list(pred_mask.squeeze(1).to("cpu").numpy())
                
                # The training data is a binary mask with values of [0, 1]
                # The sigmoid in the output layer will produce a value between [0.0, 1.0]
                # Having a simple centered threshold makes most sense. 
                masks.extend([(m>=0.5)*255 for m in minibatch_masks])

                # Experimenting with other masks
                # In the end this makes no sense, because the masks are calculated
                # by looking at the output values of single images and batches
                # the loss function in training is not taking any locally information

                # Values above the mean value of each single mask are positive, this will always result in about
                # The same number of positive/negative pixels
                #masks.extend([(m>=m.mean())*255 for m in minibatch_masks])



    original_images = []
    for i in range(3):
        # Convert from tensor to numpy [C, X, Y] -> [X, Y, C]
        original_images.append(np.moveaxis(data.dataset[i].numpy(), 0, -1))

    fig = create_figure_of_image_mask_pairs(list(zip(original_images, masks))) 
    fig.show()
    plt.show()
    
    #fig = create_figure_of_image_mask_pairs(list(zip(original_images, masks2))) 
    #fig.show()
    #plt.show()


def getLogger(module_name: str, filename: str, stdout_log_level: str) -> logging.Logger:
    format = "%(asctime)s [%(name)s:%(levelname)s] %(message)s"
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        filemode="a",
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(module_name)

    # Add handler for stdout
    if stdout_log_level:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout_log_level)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", count=True, show_default=True)
@click.option("-l", "--logfile", default=f"unet.log", show_default=True)
@click.option("-w", "--workdir", default="WD", show_default=True, type=click.Path(file_okay=False))
@click.option("-ow", "--overwrite", is_flag=True, help="Overwrite working directory")
@click.option("-d", "--device", default="cuda", show_default=True)
@click.pass_context
def unet(ctx, verbose: int, logfile: str, workdir: str, overwrite: bool, device: str):
    global log
    dev, amp = select_device(device)
    
    workdir = os.path.abspath(workdir)
    if os.path.exists(workdir):
        if os.listdir(workdir) != [] and not overwrite:
            print(f"Warning working dir {workdir} contains data! Proceed? [y/[n]]")
            selection = input()
            if selection.lower() != 'y':
                print('Aborting ...')
                exit()
    else:
        print(f"creating new working directory!")
        os.mkdir(workdir)

    ctx.obj["dev"] = dev
    ctx.obj["amp"] = amp
    ctx.obj["workdir"] = workdir
    loglevel = logging.WARNING
    if verbose > 1:
        loglevel = logging.INFO
    if verbose >= 2:
        loglevel = logging.DEBUG
    log = getLogger(
        __name__, Path(ctx.obj["workdir"]) / logfile, stdout_log_level=loglevel
    )


@unet.command()
@click.argument("data_folder")
@click.option("-e", "--epochs", default=5, show_default=True)
@click.pass_context
def train(ctx, data_folder: str, epochs: int):
    """Train a model on the given dataset folder

    Args:
        data_folder (str): File path to a folder containing a complete training & validation dataset 
    """
    _train(data_folder, epochs, ctx.obj["workdir"], ctx.obj["dev"], ctx.obj["amp"])


@unet.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_folder", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def run(ctx, model_path: str, data_folder: str):
    """Run a trained model on the given data folder.

    Args:
        model_path (str): The path to model checkpoint 
        data_folder (str): The path to the folder containing training dataset
    """
    _run(model_path, data_folder, ctx.obj["dev"])


# Create a main that is used by setup.cfg as console_script entry point
def main():
    unet(obj={})


if __name__ == "__main__":
    main()
