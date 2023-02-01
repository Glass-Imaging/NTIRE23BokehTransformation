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

import time


class BokehDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None):
        self._root_folder = root_folder
        self._transform = transform

        self._source_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._target_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._source_alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.alpha_src.png")))
        self._target_alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.alpha_tgt.png")))

        file_counts = [
            len(self._source_paths),
            len(self._target_paths),
            len(self._source_alpha_paths),
            len(self._target_alpha_paths),
        ]
        if not file_counts[0] or len(set(file_counts)) != 1:
            raise ValueError(
                f"Empty or non-matching number of files in root dir: {file_counts}. Expected an equal number of source, target source-alpha and target-alpha files."
            )

    def __len__(self):
        return len(self._source_paths)

    def __getitem__(self, index):
        source = Image.open(self._source_paths[index])
        target = Image.open(self._target_paths[index])
        source_alpha = Image.open(self._source_alpha_paths[index])
        target_alpha = Image.open(self._target_alpha_paths[index])

        if self._transform:
            source = self._transform(source)
            target = self._transform(target)
            source_alpha = self._transform(source_alpha)
            target_alpha = self._transform(target_alpha)

        return {
            "source": source,
            "target": target,
            "source_alpha": source_alpha,
            "target_alpha": target_alpha,
        }
