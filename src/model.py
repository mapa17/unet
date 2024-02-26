# from albumentations.augmentations.functional import scale
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def centercrop(x: torch.Tensor, s: torch.Size) -> torch.Tensor:
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.dconv(x)


class Upstep(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super().__init__()
        self.unconv = nn.ConvTranspose2d(
            inputChannels, outputChannels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(inputChannels, outputChannels)

    def forward(self, x : torch.Tensor, skip_tensor : torch.Tensor) -> torch.Tensor:
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
        output_channels = 1 # Only distinguish between foreground and background

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
        self.flatten = nn.Conv2d(in_channels, output_channels, kernel_size=1)

        self.output = nn.Sigmoid()
        # Output size 320, 464

        log.info(f"UNET model {self} ...")

    def forward(self, x : torch.Tensor) -> torch.Tensor:
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
            x = step(x, skip_tensor)
            log.debug(f"Decoder Step [{i}]: output tensor shape {x.shape}")

        x = self.flatten(x)
        x = self.output(x)

        log.debug(f"Final tensor: {x.shape}")
        return x


