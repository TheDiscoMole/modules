import torch

from .attention import *

class SubWordToSentenceEncoder (torch.nn.Module):
    def __init__ (self, embedding,
        attention_heads=8,
        dropout=1e-1,
        embeddings_file=None,
        layers=1
    ):
        super(TextEncoder, self).__init__()

        # validation
        assert len(embeddings.shape) == 2, "embeddings tensor must have 2 dimensions"
        assert all(size == expected for size,expected in zip(embeddings.shape,shape)), "embeddings shape doesn't match definition"

        # configs
        self.features = shape[-1]

        # modules
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self.encoders = torch.nn.ModuleList([
            AttentionEncoder(
                MultiHeadAttention(self.features, attention_heads, dropout=dropout),
                shape[-1],
                AttentionInputNd(self.features, self.features, dropout=dropout),
                AttentionInputNd(self.features, self.features, dropout=dropout),
                AttentionInputNd(self.features, self.features, dropout=dropout),
                dropout=dropout)
            for i in range(layers)])


    def forward (self, text):
        # validation
        assert ids.dims() == 2, "input must have 2 dimensions"

        # embedding
        encoding = self.embeddings(ids)

        # encoding
        for i,encoder in enumerate(self.encoders):
            encoding = self.encoder(encoding, encoding, encoding)

        # sentence
        encoding = torch.sum(encoding, dim=1)

        return encoding
