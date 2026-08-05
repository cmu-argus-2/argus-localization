"""Training pair dataset: (query image, matched satellite tile image, cluster id).

Resize/crop happens here (in DataLoader worker processes, PIL); ImageNet
normalization happens later, batched on GPU, mirroring the pattern already
used in retrievers/earthloc_retriever.py.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from core.types import GeoTile

IMAGE_SIZE = 224


def _load_resized(path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()  # (3,H,W) uint8


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[GeoTile, GeoTile]],
        cluster_ids: list[int],
        image_size: int = IMAGE_SIZE,
    ):
        assert len(pairs) == len(cluster_ids)
        self.pairs = pairs
        self.cluster_ids = cluster_ids
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        query, tile = self.pairs[idx]
        q_img = _load_resized(query.image_path, self.image_size)
        t_img = _load_resized(tile.image_path, self.image_size)
        return q_img, t_img, self.cluster_ids[idx]
