import torch

from .attention import *

class TextEncoder (torch.nn.Module):
    def __init__ (self, embedder, encoder):
        torch.nn.Module.__init__(self)

        # modules
        self.embedder = embedder
        self.encoder = encoder

    def forward (self, text):
        # encoding
        encoding = self.embedder(text)
        encoding = self.encoder(encoding, encoding, encoding)
        encoding = torch.mean(encoding, dim=1)

        return encoding
