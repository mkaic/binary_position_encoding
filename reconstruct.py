from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.io import write_jpeg
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from bin_pe import get_binary_position_encoding

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

WIDTH = 32
DEPTH = 8
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
ITERATIONS = 2000
LR = 0.01
SAVE = False

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)


class Reconstructor(nn.Module):
    def __init__(self, width, depth, pe_dim):
        super().__init__()
        layer_dims = [pe_dim] + [width] * (depth - 1) + [3]
        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(layer_dims, layer_dims[1:])):
            layers.append(nn.Linear(dim_in, dim_out))

            if i != (depth - 1):
                layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, pos_enc) -> torch.Tensor:
        x = self.mlp(pos_enc)
        x = torch.sigmoid(x)
        x = x.permute(2, 0, 1)
        return x


image = Image.open("branos.jpg").convert("RGB")
image = to_tensor(image)
write_jpeg((image * 255).to(torch.uint8), "recon/images/original.jpg")
image = image.to(DEVICE)

c, h, w = image.shape

pos_enc = get_binary_position_encoding(
    shape=(h, w),
    device=DEVICE,
)

reconstructor = Reconstructor(WIDTH, DEPTH, pos_enc.shape[-1]).to(DEVICE)
reconstructor = torch.compile(reconstructor)
optimizer = torch.optim.Adam(
    reconstructor.parameters(),
    lr=LR,
)

num_params = sum([p.numel() for p in reconstructor.parameters()])

print(f"{num_params:,} trainable parameters")
print(f"{num_params * 4 / 1024:.2f} kB")

pbar = tqdm(range(ITERATIONS + 1))

for i in pbar:
    optimizer.zero_grad()
    output = reconstructor(pos_enc)
    error = output - image
    ms_ssim_loss = -1 * ms_ssim(
        output.unsqueeze(0), image.unsqueeze(0), data_range=1, size_average=True
    )
    mse = torch.mean(torch.square(error))
    mae = torch.mean(torch.abs(error))
    loss = mse + mae + ms_ssim_loss
    loss.backward()
    optimizer.step()

    pbar.set_description(
        f"RMSE: {torch.sqrt(mse).item():.4f} | MAE: {mae.item():.4f} | MS-SSIM: {ms_ssim_loss.item():.4f}"
    )

    if i % 10 == 0:
        output = output * 255
        output = output.to("cpu", torch.uint8)
        write_jpeg(output, f"recon/images/{i:04d}.jpg")
        write_jpeg(output, f"recon/images/latest.jpg")

if SAVE:
    torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
