import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_encoders import *


class Reconstructor(nn.Module):
    def __init__(self, shape, hidden_dim, num_layers, pe_type, activation_type, device):
        super().__init__()

        self.pe_type = pe_type
        match self.pe_type:
            case "binary":
                self.pos_enc = get_binary_position_encoding(
                    shape=shape,
                    device=device,
                )
            case "sinusoidal":
                self.pos_enc = get_sinusoidal_position_vectors(
                    shape=shape,
                    num_frequencies=hidden_dim // 4,
                    device=device,
                )
            case "dumb":
                self.pos_enc = dumb_coordinate_position_encoding(
                    shape=shape,
                    device=device,
                )

        self.activation_type = activation_type
        match self.activation_type:
            case "gelu":
                self.ActivationClass = nn.GELU
                self.LinearClass = nn.Linear
            case "sinusoidal":
                self.ActivationClass = SinActivation
                self.LinearClass = SirenLinear

        pe_dim = self.pos_enc.shape[-1]

        self.in_layer = self.LinearClass(pe_dim, hidden_dim, first=True)
        self.in_activation = self.ActivationClass()

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(self.LinearClass(hidden_dim, hidden_dim))
            self.layers.append(self.ActivationClass())

        self.out_layer = nn.Linear(hidden_dim, 3)

    def evaluate(self, pos_enc) -> torch.Tensor:

        x = self.in_layer(pos_enc)

        for layer in self.layers:
            x = layer(x)

        x = self.out_layer(x)

        if self.activation_type == "sinusoidal":
            x = (x + 1) / 2
        else:
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
