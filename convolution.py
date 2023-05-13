import numpy
import torch

from .reshape import *

"""
dimension abstraction wrapper for convolutional layers

dims: number of model dimensions
in_channels: number of input channels
out_channels: number of output channels
kernel_size: size of the convolutional kernel                                   (optional|default: 5)
stride: convolutional stride length                                             (optional|default: 1)
padding: input padding scheme                                                   (optional|default: "same")
"""
def ConvNd (dims, in_channels, out_channels,
    kernel_size=5,
    padding="same",
    stride=1
):
    assert (dims > 0) and (dims < 4), "invalid number of convolution dimensions"

    # choose convolution
    match dims:
        case 1: convolution = torch.nn.Conv1d
        case 2: convolution = torch.nn.Conv2d
        case 3: convolution = torch.nn.Conv3d

    return convolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding if stride == 1 else 0)

class ConvNdFeatureEncoder (torch.nn.Module):
    """
    n-Step wise convolutional feature dimension encoder.

    dims: number of model dimensions
    in_channels: number of input channels                                       (optional|default: 3)
    out_channels: number of output channels                                     (optional|default: 64)
    layers: number of layers to achieve out_channel size                        (optional|default: 64)
    kernel_size: size of the convolutional kernel                               (optional|default: 5)
    stride: convolutional stride length                                         (optional|default: 1)
    padding: input padding scheme                                               (optional|default: "same")
    """
    def __init__ (self, dims, in_channels, out_channels,
        activation=torch.nn.GELU,
        kernel_size=5,
        padding="same",
        layers=1,
        stride=1
    ):
        super(ConvNdFeatureEncoder, self).__init__()

        # configs
        self.dims = dims

        # reshaping
        self.reshape_convolution = ReshapeSimple("b x ... c -> b c ... x")

        # modules
        convolutions = []
        channels = numpy.linspace(start=in_channels, stop=out_channels, num=layers+1, dtype=int)

        for in_channels,out_channels in zip(channels,channels[1:]):
            convolutions.append(ConvNd(dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))
            if activation is not None: convolutions.append(activation())

        self.convolution = torch.nn.Sequential(*convolutions)

    def forward (self, input):
        # validation
        assert input.dim() == self.dims + 2, f"input should have {self.dims + 2} dimensions"

        # convolution
        convolution = self.reshape_convolution(input)
        convolution = self.convolution(convolution)
        convolution = self.reshape_convolution.inverse(convolution)

        return convolution
