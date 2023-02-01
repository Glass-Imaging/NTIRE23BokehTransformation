import os

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor

from examples.dataset import BokehDataset
from examples.metrics import calculate_lpips, calculate_psnr, calculate_ssim

to_tensor = ToTensor()
to_pil = ToPILImage()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self._conv0 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self._conv1 = nn.Conv2d(16, 16, kernel_size=3, padding="same")
        self._conv2 = nn.Conv2d(16, 3, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv0(x))
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        return x


def fake_eval(model, dataloader):
    model.eval()

    batch = next(iter(dataloader))
    source = batch["source"].cuda()
    target = batch["target"].cuda()

    with torch.no_grad():
        output = model(source)

    # Calculate metrics
    lpips = np.mean([calculate_lpips(img0, img1) for img0, img1 in zip(output, target)])
    psnr = np.mean(
        [calculate_psnr(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
    )
    ssim = np.mean(
        [calculate_ssim(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
    )

    # Save images
    output = output.detach().cpu()
    for i, image in enumerate(output):
        image = to_pil(image)
        image.save(f"outputs/image_{i}.jpg")

    print(f"Metrics: lpips={lpips:0.02f}, psnr={psnr:0.02f}, ssim={ssim:0.02f}")

    model.train()


def train():
    os.makedirs("outputs", exist_ok=True)

    model = Model()
    model.train()
    model.cuda()

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    dataset = BokehDataset("./data", transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(15):
        for i_batch, batch in enumerate(dataloader):
            source = batch["source"].cuda()
            target = batch["target"].cuda()

            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} completed, loss: {loss.item():0.03f}.")
        fake_eval(model, dataloader)


if __name__ == "__main__":
    train()
