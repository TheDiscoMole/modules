import numpy
import torch

from .reshape import *

class ConvNd (torch.nn.Module):
    """
    dimension abstraction wrapper for convolutional layers

    dims: number of input dimensions
    in_channels: number of input channels
    out_channels: number of output channels
    kernel_size: size of the convolutional kernel                               (optional|default: 5)
    padding: input padding scheme                                               (optional|default: "same")
    stride: convolutional stride length                                         (optional|default: 1)
    """
    def __init__ (self, dims, in_channels, out_channels,
        kernel_size=5,
        padding="same",
        stride=1
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (dims > 0) and (dims < 4), "invalid number of convolution dimensions"

        # configs
        self.dims = dims
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        # modules
        match dims:
            case 1: convolution = torch.nn.Conv1d
            case 2: convolution = torch.nn.Conv2d
            case 3: convolution = torch.nn.Conv3d

        self.convolution = convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if stride == 1 else 0)

    def forward (self, input):
        # convolution
        return self.convolution(input)

class ConvNdFeatureEncoder (torch.nn.Module):
    """
    n-Step wise convolutional feature dimension encoder.

    dims: number of model dimensions
    in_channels: number of input channels
    out_channels: number of output channels
    activation: convolutional activation function                               (optional|default: GELU)
    dropout: dropout rate                                                       (optional|default: 0.1)
    kernel_size: size of the convolutional kernel                               (optional|default: 5)
    layers: number of layers to achieve out_channels size                       (optional|default: 1)
    padding: input padding scheme                                               (optional|default: "same")
    stride: convolutional stride length                                         (optional|default: 1)
    """
    def __init__ (self, dims, in_channels, out_channels,
        activation=torch.nn.GELU,
        dropout=1e-1,
        kernel_size=5,
        layers=1,
        padding="same",
        stride=1
    ):
        torch.nn.Module.__init__(self)

        # configs
        self.dims = dims

        # reshaping
        self.reshape_convolution = ReshapeSimple("b x ... c -> b c ... x")

        # modules
        dropout = torch.nn.Dropout(dropout)

        convolutions = numpy.linspace(start=in_channels, stop=out_channels, num=layers+1, dtype=int)
        convolutions = [
            ConvNd(
                dims=dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride)
            for in_channels,out_channels in zip(convolutions,convolutions[1:])]

        layers = zip([dropout] * len(convolutions), convolutions) if activation is None else zip([dropout] * len(convolutions), convolutions, [activation()] * len(convolutions))
        layers = [element for layer in layers for element in layer]

        self.layers = torch.nn.ModuleList(layers[1:])

    def forward (self, input):
        # validation
        assert input.dim() == self.dims + 2, f"input should have {self.dims + 2} dimensions"

        # convolution
        convolution = self.reshape_convolution(input)
        for layer in self.layers: convolution = layer(convolution)
        convolution = self.reshape_convolution.inverse(convolution)

        return convolution
