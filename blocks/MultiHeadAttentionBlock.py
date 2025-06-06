import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        """
        Initialization of multi head attention.
`
        args:
            d_in: dimension of input vector
            d_out: dimension of output vector
            context_length
            dropout: dropout rate, 0.0 indicates no dropout
            num_heads
            qkv_bias: whether to use bias or not
        """

        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )
        
    def forward(self, x):
        """
        Forward method of multi head attention.

        args:
            x: tensor of shape (batch, num_tokens, d_in)

        returns:
            tensor of shape (batch, num_tokens, d_out)
        """
        batch_size, num_tokens, _ = x.shape
        # shape: (batch_size, num_tokens, d_in)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # shape: (batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # shape: (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # get the mask matrix
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # apply mask_bool to attn_scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        
        # shape: (batch_size, num_heads, num_tokens, head_dim)
        # -> (batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # shape: (batch_size, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec