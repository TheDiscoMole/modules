import torch

from .shape import ReshapePatchesNdFeatures

def MaxPoolNd(dims: int) -> torch.nn.Module:
    match dims:
        case 1: return torch.nn.MaxPool1d
        case 2: return torch.nn.MaxPool2d
        case 3: return torch.nn.MaxPool3d
        case _: raise NotImplementedError(f"MaxPool not implemented for dims={dims}")

def ConvTransposeNd(dims: int):
    match dims:
        case 1: return torch.nn.ConvTranspose1d
        case 2: return torch.nn.ConvTranspose2d
        case 3: return torch.nn.ConvTranspose3d
        case _: raise NotImplementedError(f"ConvTransposeNd not implemented for dims={dims}")

class UpscaleNdFeatures():
    """
    Upscale n-dimensions by splitting feature space

    scaleFactor: factor with which to scale input dimensions
    dims: number of model dimensions                         (optional|default: 1)
    """
    def __init__(self, scaleFactor: int,
        dims: int=1             
    ) -> None:
        self.reshape = ReshapePatchesNdFeatures([scaleFactor for dim in range(dims)])
        self.scaleFactor = scaleFactor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.reshape.inverse(input, [input.shape[0]] + [size * self.scaleFactor for size in input.shape[1:-1]] + [input.shape[-1]])
