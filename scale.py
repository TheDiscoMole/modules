import copy
import torch

from .attention import *
from .convolution import *
from .reshape import *

class UpscaleNd (torch.nn.Module):
    """
    Upscale n-dimensional feature space.

    features: number of model features
    scale_factor: factor with which to scale input dimensions
    attention_heads: number of attention heads                                  (optional|default: 8)
    convolutions: number of convolutions                                        (optional|default: 3)
    dims: number of model dimensions                                            (optional|default: 1)
    dropout: dropout rate                                                       (optional|default: 0.1)
    kernel_size: convolutional kernel size                                      (optional|default: 5)
    mdims: number of memory dimensions                                          (optional|default: dims)
    patch_size: ndim sub-patch size                                             (optional|default: 4)
    """
    def __init__ (self, features, scale_factor,
        attention_heads=8,
        convolutions=3,
        dims=1,
        dropout=1e-1,
        kernel_size=5,
        mdims=None,
        patch_size=4
    ):
        super(UpscaleNd, self).__init__()

        # configs
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        # reshaping
        self.reshape_rescaling = ReshapePatchesNdFeatures([scale_factor for dim in range(dims)])

        # modules
        query_in_features = features // (scale_factor ** dims)
        attention_out_features = features * (patch_size ** dims) if dims > 1 else features
        patches = [patch_size] * dims if dims > 1 else [1]

        self.attention = AttentionEncoder(
            MultiHeadAttention(attention_out_features, attention_heads, dropout=dropout),
            features,
            AttentionInputNd(features, attention_out_features, dims=dims, patches=patches, dropout=dropout),
            AttentionInputNd(query_in_features, attention_out_features, dims=dims, patches=patches, dropout=dropout),
            AttentionInputNd(features, attention_out_features, dims=dims, patches=patches, dropout=dropout),
            dropout=dropout)

        self.convolution = ConvNdFeatureEncoder(dims, features, features, layers=2, kernel_size=kernel_size)

    def forward (self, input,
        memory=None,
    ):
        # validation
        assert (memory is None) or (memory.shape[-1] == input.shape[-1]), "attention query feature-dimenion must match inputs"

        # attention
        attention = input if memory is None else memory, self.reshape_rescaling.inverse(input, [input.shape[0]] + [size * self.scale_factor for size in input.shape[1:-1]] + [input.shape[-1]]), input if memory is None else memory
        attention = self.attention(*attention)

        # convolution
        convolution = self.convolution(attention)

        return convolution
