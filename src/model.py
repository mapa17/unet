import torch
import torch.nn as nn

def centercrop(x: torch.tensor, s: torch.Size) -> torch.tensor:
    """
    Apply center cropping on given tensor `x`, so that the cropped
    tensor has the size defined by `s`.
    """

    # Calculate by how much to crop. If the target size is bigger
    # than the tensor, do nothing.
    h = x.shape[-2]
    w = x.shape[-1]
    
    hc = h - s[-2]
    wc = w - s[-1]
    if (wc < 0) or (hc < 0):
        return x

    hc = hc // 2
    wc = wc // 2

    xcropped = x[:, :, hc:h-hc, wc:w-wc]
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
            nn.Conv2d(inputChannels, outputChannels, kernel_size, bias=False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputChannels, outputChannels, kernel_size, bias=False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.dconv(x)

class Upstep(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super().__init__()
        self.unconv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=2, stride=2)
        self.conv = DoubleConv(inputChannels, outputChannels)

    def forward(self, x, x2):
        """
        Do:
        * Unconv to x
        * centercrop x2 to (unconv) x
        * concatenate the two
        * double conv on the concat
        """
        x = self.unconv(x)
        x2 = centercrop(x2, x.shape)
        x = torch.cat((x, x2), dim=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        convolution_channels = [64, 128, 256, 512]
        input_channel = 1
        output_channels = 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_channels = input_channel
        for channels in convolution_channels:
            self.encoder.append(DoubleConv(in_channels, channels))
            in_channels = channels

        self.bottleneck = DoubleConv(in_channels, in_channels*2) 
    
        in_channels = in_channels*2
        for channels in reversed(convolution_channels):
            self.decoder.append(Upstep(in_channels, channels))
            in_channels = channels
 
        self.output = nn.Conv2d(in_channels, output_channels, kernel_size=3)

        """       
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(output_channels))
        """

    def forward(self, x):
        encoded_tensors = []

        # Downwards path
        for i, step in enumerate(self.encoder):
            print(f"Encoder Step [{i}]: input tensor shape {x.shape}")
            x = step(x)
            encoded_tensors.append(x)
            x = self.pool(x)
            print(f"Encoder Step [{i}]: output tensor shape {x.shape}")
        
        # Bottleneck
        print(f"Bottleneck: input tensor shape {x.shape}")
        x = self.bottleneck(x)
        print(f"Bottleneck: output tensor shape {x.shape}")

        # Upwards path
        for i, (step, enc_tensor) in enumerate(zip(self.decoder, reversed(encoded_tensors))):
            print(f"Decoder Step [{i}]: input tensor shape {x.shape}")
            x = step(x, enc_tensor)
            print(f"Decoder Step [{i}]: output tensor shape {x.shape}")
        
        x = self.output(x)

        print(f"Final tensor: {x.shape}")
        return x
 