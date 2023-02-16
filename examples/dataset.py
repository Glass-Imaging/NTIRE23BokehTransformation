import glob
import os.path as osp
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class BokehDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None):
        self._root_folder = root_folder
        self._transform = transform

        self._source_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._target_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.alpha.png")))

        self._meta_data = self._read_meta_data(osp.join(root_folder, "meta.txt"))

        file_counts = [
            len(self._source_paths),
            len(self._target_paths),
            len(self._alpha_paths),
            len(self._meta_data),
        ]
        if not file_counts[0] or len(set(file_counts)) != 1:
            raise ValueError(
                f"Empty or non-matching number of files in root dir: {file_counts}. "
                "Expected an equal number of source, target, source-alpha and target-alpha files. "
                "Also expecting matching meta file entries."
            )

    def __len__(self):
        return len(self._source_paths)

    def _read_meta_data(self, meta_file_path: str):
        """Read the meta file containing source / target lens and disparity for each image.

        Args:
            meta_file_path (str): File path

        Raises:
            ValueError: File not found.

        Returns:
            dict: Meta dict of tuples like {id: (id, src_lens, tgt_lens, disparity)}.
        """
        if not osp.isfile(meta_file_path):
            raise ValueError(f"Meta file missing under {meta_file_path}.")

        meta = {}
        with open(meta_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
            meta[id] = (src_lens, tgt_lens, disparity)
        return meta

    def __getitem__(self, index):
        source = Image.open(self._source_paths[index])
        target = Image.open(self._target_paths[index])
        alpha = Image.open(self._alpha_paths[index])

        filename = osp.basename(self._source_paths[index])
        id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = self._meta_data[id]

        if self._transform:
            source = self._transform(source)
            target = self._transform(target)
            alpha = self._transform(alpha)

        return {
            "source": source,
            "target": target,
            "alpha": alpha,
            "src_lens": src_lens,
            "tgt_lens": tgt_lens,
            "disparity": disparity,
        }
