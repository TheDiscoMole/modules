import numpy
import torch

from .normalize import BatchNormNd

class ConvNd (torch.nn.Module):
    """
    dimension abstraction wrapper for convolutional layers

    dims: number of input dimensions
    inChannels: number of input channels
    outChannels: number of output channels
    kernelSize: size of the convolutional kernel           (optional|default: 5)
    padding: input padding scheme                          (optional|default: "same")
    stride: convolutional stride length                    (optional|default: 1)
    """
    def __init__(self, dims: int, inChannels: int, outChannels: int,
        kernelSize: int=5,
        padding: str="same",
        stride: int=1
    ) -> None:
        super().__init__()

        # configs
        self.dims = dims
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride

        # modules
        match dims:
            case 1: convolution = torch.nn.Conv1d
            case 2: convolution = torch.nn.Conv2d
            case 3: convolution = torch.nn.Conv3d
            case _: NotImplementedError(f"ConvNd not implemented for dim={dims}")

        self.convolution = convolution(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding if stride == 1 else 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.convolution(input)

class ConvNdFeatureEncoder(torch.nn.Module):
    """
    n-Step wise convolutional feature dimension encoder.

    dims: number of model dimensions
    inChannels: number of input channels
    outChannels: number of output channels
    activation: convolutional activation function         (optional|default: GELU)
    dropout: dropout rate                                 (optional|default: 0.1)
    kernelSize: size of the convolutional kernel          (optional|default: 5)
    layers: number of layers to achieve out_channels size (optional|default: 1)
    padding: input padding scheme                         (optional|default: "same")
    stride: convolutional stride length                   (optional|default: 1)
    """
    def __init__(self, dims: int, inChannels: int, outChannels: int,
        activation: torch.nn.Module=torch.nn.GELU,
        dropout: float=1e-1,
        kernelSize: int=5,
        layers: int=1,
        normalization: torch.nn.Module=BatchNormNd,
        padding: str="same",
        stride: int=1
    ) -> None:
        super().__init__()

        # configs
        self.dims = dims
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride

        # modules
        channels = numpy.linspace(start=inChannels, stop=outChannels, num=layers+1, dtype=int)
        self.layers = []

        for inChannels,outChannels in zip(channels,channels[1:]):
            self.layers.append(
                ConvNd(
                    dims=dims,
                    inChannels=inChannels,
                    outChannels=outChannels,
                    kernelSize=kernelSize,
                    padding=padding,
                    stride=stride))
            if normalization: 
                self.layers.append(normalization(dims)(outChannels))
            if activation: 
                self.layers.append(activation())
            if dropout: 
                self.layers.append(torch.nn.Dropout(dropout))

        self.layers = torch.nn.Sequential(*self.layers[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)