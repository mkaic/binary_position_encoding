from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import write_jpeg
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from .reconstructor import *
import kornia.color as kc
import torch.nn as nn

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
gpu = parser.parse_args().gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

hidden_dim = 64
num_layers = 4
pe_type = "binary_sinusoidal"
activation_class = nn.GELU


iterations = 2000
lr = 0.01
save = False

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)

image_rgb = Image.open("jwst_cliffs.png").convert("RGB")
# image_rgb = Image.open("branos.jpg").convert("RGB")
# image_rgb = Image.open("monalisa.jpg").convert("RGB")
# image_rgb = Image.open("minion.jpg").convert("RGB")
# image_rgb = Image.open("/workspace/projects/ok.jpg").convert("RGB")

image_rgb = to_tensor(image_rgb).to(device)

write_jpeg(
    (image_rgb * 255).to("cpu", torch.uint8),
    "recon/original.jpg",
    quality=100,
)

c, h, w = image_rgb.shape

reconstructor = (
    Reconstructor(
        shape=(h, w),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pe_type=pe_type,
        activation_class=activation_class,
        device=device,
    )
    .to(device)
    .to(dtype)
)

optimizer = torch.optim.AdamW(
    reconstructor.parameters(),
    lr=lr,
)

num_params = sum([p.numel() for p in reconstructor.parameters()])

print(f"{num_params:,} trainable parameters")
print(f"{num_params * 4 / 1024 / 2:.2f} kB")

pbar = tqdm(range(iterations + 1))

for i in pbar:
    optimizer.zero_grad()
    output_rgb = reconstructor()

    error_rgb = output_rgb - image_rgb

    mse_rgb = torch.mean(torch.square(output_rgb - image_rgb))
    mae_rgb = torch.mean(torch.abs(output_rgb - image_rgb))

    mse_rgb.backward()
    optimizer.step()

    pbar.set_description(
        f"RMSE: {torch.sqrt(mse_rgb).item():.4f} | MAE: {mae_rgb.item():.4f}"  # | MS-SSIM: {ms_ssim_loss.item():.4f}"
    )

    if i % 10 == 0:
        output_rgb = output_rgb * 255
        output_rgb = output_rgb.to("cpu", torch.uint8)
        write_jpeg(
            output_rgb,
            f"recon/images/{i:04d}.jpg",
            quality=100,
        )
        write_jpeg(
            output_rgb,
            f"recon/latest.jpg",
            quality=100,
        )

if save:
    torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
