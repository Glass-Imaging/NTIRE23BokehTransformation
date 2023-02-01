from typing import Tuple, Callable, Optional, TypedDict
import sys
import glob
import os
import os.path as osp
from torch.utils.data import Dataset
import cv2
import numpy as np
import bokeh_rendering
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import center_of_mass
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.optim import Adam

from .dataset import BokehDataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self._conv0 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self._conv1 = nn.Conv2d(16, 16, kernel_size=3, padding="same")
        self._conv2 = nn.Conv2d(16, 16, kernel_size=3, padding="same")
        self._conv3 = nn.Conv2d(16, 3, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv0(x))
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))
        return x


def train():
    model = Model()
    model.train()
    model.cuda()

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.L1Loss()

    dataset = BokehDataset("./data")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    to_tensor = T.ToTensor()

    for epoch in range(3):
        for i_batch, batch in enumerate(dataloader):
            source = to_tensor(batch["source"]).cuda()
            target = to_tensor(batch["target"]).cuda()

            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()

            if i_batch % 25 == 0:
                print(f"Epoch {epoch}, batch {i_batch}: loss: {loss.item():0.03f}.")


if __name__ == "__main__":
    train()
