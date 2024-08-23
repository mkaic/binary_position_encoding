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
            case "dumb":
                self.pos_enc = get_dumb_coordinate_position_encoding(
                    shape=shape,
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

        layer_sizes = [self.pe_dim] + [self.hidden_dim * 2] * (num_layers - 1) + [3]

        self.mlp = MultPosMLP(layer_sizes, self.activation_class)

    def evaluate(self, pos_enc) -> torch.Tensor:

        x = self.mlp(pos_enc)

        x = torch.sigmoid(x)

        return x.permute(2, 0, 1)

    def forward(self):
        return self.evaluate(self.pos_enc)


class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30 * x)


class SirenLinear(nn.Module):
    def __init__(self, in_features, out_features, first=False, last=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first = first
        self.last = last
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        if self.first:
            bound = 1 / self.in_features
        else:
            bound = math.sqrt((6 / self.in_features))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


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
        chunk_size = x_original.shape[-1]

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation(x)
            if i < len(self.layers) - 2:
                x[..., :chunk_size] = x[..., :chunk_size] * x_original

            # x = x + torch.randn_like(x) * 0.1

        return x
