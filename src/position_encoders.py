import torch
import math


def get_binary_position_encoding(shape, device):

    longest_side = max(shape)
    num_frequencies = math.ceil(math.log2(longest_side))

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.int, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    # snippet to convert from int to binary is adapted from https://stackoverflow.com/questions/62828620/how-do-i-convert-int8-into-its-binary-representation-in-pytorch
    mask = 2 ** torch.arange(
        num_frequencies - 1, -1, -1, dtype=torch.int, device=device
    )
    positions = positions.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=torch.bfloat16)

    positions = positions * 2 - 1

    positions = positions.view(*shape, -1)

    return positions


def get_sinusoidal_position_encoding(shape, num_frequencies, device):

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.bfloat16, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    freq_bands = []

    for freq_idx in range(1, num_frequencies + 1):
        for pe_axis in range(len(shape)):

            pos = positions[..., pe_axis] * (
                1 / (10000 ** (freq_idx / num_frequencies))
            )

            cos = torch.cos(pos)
            sin = torch.sin(pos)

            freq_bands.append(cos)
            freq_bands.append(sin)

    positions = torch.stack(freq_bands, dim=-1)  # H, W, C

    return positions


def get_dumb_coordinate_position_encoding(shape, device):
    positions = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(i, dtype=torch.bfloat16, device=device) for i in shape
            ],  # [0, 1]
            indexing="ij"
        ),
        dim=-1,
    )

    return positions


def get_binary_sinusoidal_position_encoding(shape, device):

    longest_side = max(shape)
    num_frequencies = math.ceil(math.log2(longest_side))

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.bfloat16, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    freq_bands = []

    for freq_idx in range(1, num_frequencies + 1):
        for pe_axis in range(len(shape)):

            pos = positions[..., pe_axis] / (2**freq_idx)

            cos = torch.cos(pos)
            sin = torch.sin(pos)

            freq_bands.append(cos)
            freq_bands.append(sin)

    positions = torch.stack(freq_bands, dim=-1)  # H, W, C

    return positions
