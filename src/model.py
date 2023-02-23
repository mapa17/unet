# from albumentations.augmentations.functional import scale
import logging

from dataset import CaravanImageDataLoader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.profiler
from torch.profiler import tensorboard_trace_handler

from tqdm import tqdm


def centercrop(x: torch.tensor, s: torch.Size) -> torch.tensor:
    """
    Apply center cropping on given tensor `x`, so that the cropped
    tensor has the size defined by `s`.
    """
    log.debug(f"centercrop {x.shape} -> {s}")
    if x.shape != s:
        # Calculate by how much to crop. If the target size is bigger
        # than the tensor, do nothing.
        hc = x.shape[-2] - s[-2]
        wc = x.shape[-1] - s[-1]

        if (wc < 0) or (hc < 0):
            return x

        # How much to crop from each side
        hc = hc // 2
        wc = wc // 2

        xcropped = x[:, :, hc : hc + s[-2], wc : wc + s[-1]]
    else:
        xcropped = x

    return xcropped


class DoubleConv(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        """
        Create a sequence of two double convolutions using ReLU
        and add BatchNormalization (not part of the original paper)
        """
        super().__init__()
        kernel_size = 3
        self.dconv = nn.Sequential(
            nn.Conv2d(
                inputChannels,
                outputChannels,
                kernel_size,
                stride=1,
                padding=1,
                bias=False, # Bias can be set to False because we follow with  BatchNorm which removes the bias when subtracting the mean
            ),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outputChannels,
                outputChannels,
                kernel_size,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dconv(x)


class Upstep(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super().__init__()
        self.unconv = nn.ConvTranspose2d(
            inputChannels, outputChannels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(inputChannels, outputChannels)

    def forward(self, x, skip_tensor):
        """
        Do:
        * Unconv to x
        * centercrop x2 to (unconv) x
        * concatenate the two
        * double conv on the concat
        """
        x = self.unconv(x)
        skip_tensor = centercrop(skip_tensor, x.shape)
        x = torch.cat((x, skip_tensor), dim=1)
        x = self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        convolution_channels = [64, 128, 256, 512]
        input_channel = 3
        output_channels = 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = input_channel
        for channels in convolution_channels:
            self.encoder.append(DoubleConv(in_channels, channels))
            in_channels = channels

        self.bottleneck = DoubleConv(in_channels, in_channels * 2)

        in_channels = in_channels * 2
        for channels in reversed(convolution_channels):
            self.decoder.append(Upstep(in_channels, channels))
            in_channels = channels

        # Each segmentation label has its own output channel!
        self.output = nn.Conv2d(in_channels, output_channels, kernel_size=1)

        log.info(f"Created model with parameters {self.parameters()}")
        """
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(output_channels))
        """

    def forward(self, x):
        skip_tensors = []

        # Downwards path
        for i, step in enumerate(self.encoder):
            log.debug(f"Encoder Step [{i}]: input tensor shape {x.shape}")
            x = step(x)
            skip_tensors.append(x)
            x = self.pool(x)
            log.debug(f"Encoder Step [{i}]: output tensor shape {x.shape}")

        # Bottleneck
        log.debug(f"Bottleneck: input tensor shape {x.shape}")
        x = self.bottleneck(x)
        log.debug(f"Bottleneck: output tensor shape {x.shape}")

        # Upwards path
        for i, (step, skip_tensor) in enumerate(
            zip(self.decoder, reversed(skip_tensors))
        ):
            log.debug(
                f"Decoder Step [{i}]: input tensor shape {x.shape}, skip tensor shape {skip_tensor.shape}"
            )
            # from pudb import set_trace as st; st()
            x = step(x, skip_tensor)
            log.debug(f"Decoder Step [{i}]: output tensor shape {x.shape}")

        x = self.output(x)
        log.debug(f"Final tensor: {x.shape}")
        return x


class ModelManager:
    def __init__(
        self,
        model: nn.Module,
        data: CaravanImageDataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        working_dir: str,
        dev: str,
        amp: bool,
    ):
        self._model = model
        self._training_data = data.training_loader
        self._validation_data = data.validation_loader
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._working_dir = working_dir
        self._dev = dev
        self.amp = amp

        if self.amp:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

    def train_epoch(self):
        loss = 0

        #with torch.profiler.profile(
        #    schedule=torch.profiler.schedule(
        #        wait=1,
        #        warmup=1,
        #        active=6,
        #        repeat=1),
        #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard'),
        #    with_stack=True
        #) as profiler:
        self._model.train()
        with tqdm(self._training_data, desc="TRN") as loop:
            for batch_idx, (data, targets) in enumerate(loop):
                log.debug(f"Training batch {batch_idx}/{len(self._training_data)} ...")
                data = data.to(device=self._dev)
                targets = targets.float().unsqueeze(1).to(device=self._dev)

                # If automatic mixed precision is available, use it with the GradScaler to increase performance
                if self.amp:
                    # Forward
                    with torch.amp.autocast(
                        device_type=self._dev, dtype=torch.bfloat16
                    ):
                        predictions = self._model(data)
                        loss = self._loss_fn(predictions, targets)

                    # backward
                    self._optimizer.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    # Forward
                    predictions = self._model(data)
                    loss = self._loss_fn(predictions, targets)

                    # backward
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self._optimizer.step()

                # update tqdm loop
                loop.set_postfix(loss=loss.item())
                #profiler.step()

        logging.debug("Running validation ...")
        val_loss = 0
        self._model.eval()
        with torch.no_grad():
            with tqdm(self._validation_data, desc="VAL") as loop:
                for batch_idx, (data, targets) in enumerate(loop):
                    self._optimizer.zero_grad()
                    data = data.to(device=self._dev)
                    targets = targets.float().unsqueeze(1).to(device=self._dev)

                    # forward
                    predictions = self._model(data)
                    val_loss += self._loss_fn(predictions, targets)

                    loop.set_postfix(loss=val_loss.item()/(batch_idx+1))

        val_loss = val_loss / len(self._validation_data)
        return loss, val_loss

    def train(self):
        pass
