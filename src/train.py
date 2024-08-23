from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.io import write_jpeg
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from .reconstructor import Reconstructor

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
gpu = parser.parse_args().gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

hidden_dim = 64
num_layers = 8
pe_type = "dumb"
activation_type = "sinusoidal"


iterations = 2000
lr = 0.0001
save = False

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)

image = Image.open("jwst_cliffs.png").convert("RGB")
image = to_tensor(image).to(device)
write_jpeg(
    (image * 255).to("cpu", torch.uint8),
    "recon/images/original.jpg",
    quality=100,
)

c, h, w = image.shape

reconstructor = Reconstructor(
    shape=(h, w),
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    pe_type=pe_type,
    activation_type=activation_type,
    device=device,
).to(device)

optimizer = torch.optim.AdamW(
    reconstructor.parameters(),
    lr=lr,
)

num_params = sum([p.numel() for p in reconstructor.parameters()])

print(f"{num_params:,} trainable parameters")
print(f"{num_params * 4 / 1024:.2f} kB")

pbar = tqdm(range(iterations + 1))

for i in pbar:
    optimizer.zero_grad()
    output = reconstructor()
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
        write_jpeg(
            output,
            f"recon/images/{i:04d}.jpg",
            quality=100,
        )
        write_jpeg(
            output,
            f"recon/images/latest.jpg",
            quality=100,
        )

if save:
    torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
