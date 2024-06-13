import torch

from .shape import ReshapeSimple

class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module implementation.

    features: feature size of the attention mechanism
    heads: number of attention heads
    dropout: dropout rate                             (optional|default: 0.1)
    """
    def __init__(self, features: int, heads: int,
        dropout: float=1e-1
    ) -> None:
        super().__init__()

        # validation
        assert features % heads == 0, "features needs to be divisible by heads"

        # configs
        self.dropout = dropout
        self.features = features
        self.heads = heads

        # reshaping
        self.reshapeHeads = ReshapeSimple("b ... (f h) -> b h ... f")

        # modules
        self.normalize = torch.nn.LayerNorm(features)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, 
        isCausal: bool=False, 
        queryMask: torch.Tensor|None=None
    ) -> torch.Tensor:
        # validation
        assert query.dim() == key.dim() == value.dim() == 3, f"query, key and value must have 3 dimensions"
        assert query.shape[0] == key.shape[0] == value.shape[0], "query, key and value must have same batches"
        assert key.shape[-2] == value.shape[-2], "key and value must have same sequence length"
        assert query.shape[-1] == key.shape[-1] == value.shape[-1] == self.features, f"query, key and value must have {self.features} features"

        if queryMask is not None: assert query.shape == queryMask.shape, "query mask doesn't fit query shape"

        # inputs
        input = self.reshapeHeads(query, h=self.heads), self.reshapeHeads(key, h=self.heads), self.reshapeHeads(value, h=self.heads)

        # attention
        attention = torch.nn.functional.scaled_dot_product_attention(*input, attn_mask=queryMask, dropout_p=self.dropout if self.training else 0, is_causal=isCausal)
        attention = self.reshapeHeads.inverse(attention, h=self.heads)

        # residual
        residual = attention + query
        residual = self.normalize(residual)

        return residual

class AttentionEncoder(torch.nn.Module):
    """
    general attention encoder

    key: key input module
    query: query input module
    value: value input module
    attention: attention mechanism module
    output: output module
    activation: activation function       (optional|default: sigmoid)
    dropout: dropout rate                 (optional|default: 0.1)
    """
    def __init__(self, key: torch.nn.Module, query: torch.nn.Module, value: torch.nn.Module, attention: torch.nn.Module, output: torch.nn.Module,
        activation: torch.nn.Module=torch.nn.Sigmoid,
        dropout: float=1e-1
    ) -> None:
        super().__init__()

        # modules
        self.dropout = torch.nn.Dropout(dropout)

        self.key = key
        self.query = query
        self.value = value

        self.attention = attention

        self.output = output
        self.activation = activation() if activation is not None else None

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # attention
        attention = self.key(key), self.query(query), self.value(value)
        attention = self.attention(*attention)

        # output
        output = self.output(attention)
        output = self.activation(output)

        return output
