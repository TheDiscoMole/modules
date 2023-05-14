import functools
import torch

from .convolution import *
from .reshape import *

class MultiHeadAttention (torch.nn.Module):
    """
    Multi-Head Attention module implementation.

    features: feature size of the attention mechanism
    heads: number of attention heads
    dropout: dropout rate                                                       (optional|default: 0.1)
    """
    def __init__ (self, features, heads,
        dropout=1e-1
    ):
        super(MultiHeadAttention, self).__init__()

        # validation
        assert features % heads == 0, "features needs to be divisible by heads"

        # configs
        self.dropout = dropout
        self.features = features
        self.heads = heads

        # reshaping
        self.reshape_heads = ReshapeSimple("b ... (f h) -> b h ... f")

        # modules
        self.normalize = torch.nn.LayerNorm(features)

    def forward (self, key, query, value,
        is_causal=False,
        query_mask=None
    ):
        # validation
        assert query.dim() == key.dim() == value.dim() == 3, f"query, key and value must have 3 dimensions"
        assert query.shape[0] == key.shape[0] == value.shape[0], "query, key and value must have same batches"
        assert key.shape[-2] == value.shape[-2], "key and value must have same sequence length"
        assert query.shape[-1] == key.shape[-1] == value.shape[-1] == self.features, f"query, key and value must have {self.features} features"

        if is_causal: assert attention_mask is not None, "cannot be causal without query mask"
        if query_mask is not None: assert query.shape == query_mask.shape, "query mask doesn't fit query shape"

        # inputs
        input = self.reshape_heads(query, h=self.heads), self.reshape_heads(key, h=self.heads), self.reshape_heads(value, h=self.heads)

        # attention
        attention = torch.nn.functional.scaled_dot_product_attention(*input, attn_mask=query_mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        attention = self.reshape_heads.inverse(attention, h=self.heads)

        # residual
        residual = attention + query
        residual = self.normalize(residual)

        return residual

class AttentionEncoder (torch.nn.Module):
    """
    general attention encoder

    attention: attention mechanism module
    key: key input module
    query: query input module
    value: value input module
    activation: activation function                                             (optional|default: sigmoid)
    dropout: dropout rate                                                       (optional|default: 0.1)
    """
    def __init__ (self, attention, features, key, query, value,
        activation=torch.nn.Sigmoid,
        dropout=1e-1
    ):
        super(AttentionEncoder, self).__init__()

        # modules
        self.dropout = torch.nn.Dropout(dropout)

        self.key = key
        self.query = query
        self.value = value

        self.attention = attention

        self.linear = torch.nn.Linear(features, features)
        self.activation = activation() if activation is not None else None

    def forward (self, key, query, value,
        is_causal=False,
        query_mask=None
    ):
        # attention
        attention = self.key(key), self.query(query), self.value(value)
        attention = self.attention(*attention, is_causal=is_causal, query_mask=query_mask)

        # output
        output = self.linear(attention)
        output = self.activation(output)

        return output
