from math import pi
import torch
import torch.nn as nn


# Below is a computationally cheaper version of GELU activation function:
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

        args:
            x: tensor of shape (batch_size, num_tokens, d_in)
        """
        return 0.5 * x * (1 + torch.tanh( torch.sqrt(torch.tensor(2 / pi)) * (x + 0.044715 * x ** 3)))