import torch
import math


def get_binary_position_encoding(shape, device):

    longest_side = max(shape)
    num_frequencies = math.ceil(math.log2(longest_side))
    print(num_frequencies)

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
    positions = positions.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=torch.float32)

    positions = positions * 2 - 1

    positions = positions.view(*shape, -1)

    return positions
