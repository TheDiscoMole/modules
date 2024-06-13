import torch

def BatchNormNd(dims: int) -> torch.nn.Module:
    match dims:
        case 1: return torch.nn.BatchNorm1d
        case 2: return torch.nn.BatchNorm2d
        case 3: return torch.nn.BatchNorm3d
        case _: raise NotImplementedError(f"BatchNormNd not implemented for dims={dims}")