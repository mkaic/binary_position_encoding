import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_encoders import *


class Reconstructor(nn.Module):
    def __init__(
        self, shape, hidden_dim, num_layers, pe_type, activation_class, device
    ):
        super().__init__()

        self.shape = shape
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.pe_type = pe_type
        match self.pe_type:
            case "binary":
                self.pos_enc = get_binary_position_encoding(
                    shape=shape,
                    device=device,
                )
            case "sinusoidal":
                self.pos_enc = get_sinusoidal_position_encoding(
                    shape=shape,
                    num_frequencies=hidden_dim // 4,
                    device=device,
                )
            case "binary_sinusoidal":
                self.pos_enc = get_binary_sinusoidal_position_encoding(
                    shape=shape,
                    device=device,
                )

        self.pe_dim = self.pos_enc.shape[-1]

        self.activation_class = activation_class

        self.hidden_dim = hidden_dim

        layer_sizes = [self.pe_dim] + [self.hidden_dim] * (num_layers - 1) + [3]

        self.mlp = MultPosMLP(layer_sizes, self.activation_class)

    def evaluate(self, pos_enc) -> torch.Tensor:

        x = self.mlp(pos_enc)

        x = torch.sigmoid(x)

        return x.permute(2, 0, 1)

    def forward(self):
        return self.evaluate(self.pos_enc)


class MultPosMLP(nn.Module):
    def __init__(self, layer_sizes, activation_class):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(nn.Linear(dim_in, dim_out))
            if not i == len(layer_sizes) - 2:
                self.activations.append(activation_class())
            else:
                self.activations.append(nn.Identity())

    def forward(self, x):
        x_original = x
        input_hidden_dim = x_original.shape[-1]

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation(x)
            if i < len(self.layers) - 1:
                x[..., :input_hidden_dim] = x[..., :input_hidden_dim] * x_original

            # x = x + torch.randn_like(x) * 0.1

        return x
