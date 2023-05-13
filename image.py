import torch

from .reshape import *

class ImageUpscaler (torch.nn.Module):
    """
    - encodes an image into feature space
    - creates a sub selected on smaller image patches
    - upscales the individual patches
    - recombines the upscaled patches
    - decodes the image from the feature space

    decoder: decode the image channels
    encoder: encode the image channels
    upscaler: upscale the encoded features
    patch_size: patched image sizes                                             (optional|default: 40)
    """
    def __init__ (self, decoder, encoder, upscaler,
        patch_size=40
    ):
        super(ImageUpscaler, self).__init__()

        # configs
        self.patch_size = patch_size

        # reshaping
        self.reshape_patches = ReshapePatchesNdBatches([patch_size for i in range(2)])
        self.reshape_patches_scaled = ReshapePatchesNdBatches([patch_size * upscaler.scale_factor for i in range(2)])

        # modules
        self.encoder = encoder
        self.upscaler = upscaler
        self.decoder = decoder

    def forward (self, image):
        # validation
        assert image.dim() == 4, "image batch input must have 4 dimensions"

        # encode
        image = self.encoder(image)

        # upscale
        upscaled = self.reshape_patches(image)
        upscaled = self.upscaler(upscaled)
        upscaled = self.reshape_patches_scaled.inverse(upscaled, [image.shape[0]] + [size * self.upscaler.scale_factor for size in image.shape[1:-1]] + [image.shape[-1]])

        # decode
        image = self.decoder(upscaled)

        return image
