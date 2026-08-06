"""Training pair dataset: (query image, matched satellite tile image, index).

Resize/crop happens here (in DataLoader worker processes, PIL); ImageNet
normalization happens later, batched on GPU, mirroring the pattern already
used in retrievers/earthloc_retriever.py.

Deliberately returns the pair's own index rather than looking up its cluster
id here: with num_workers>0 (persistent_workers especially), each worker
holds its own forked copy of this Dataset, so mutating a cluster-id list on
the main-process Dataset object (as periodic reclustering needs to do, see
train.py's --recluster-every-steps) would silently never reach the workers.
Returning the index instead lets the training loop look up (and update)
cluster ids purely in the main process, sidestepping that entirely.
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
    def __init__(self, pairs: list[tuple[GeoTile, GeoTile]], image_size: int = IMAGE_SIZE):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        query, tile = self.pairs[idx]
        q_img = _load_resized(query.image_path, self.image_size)
        t_img = _load_resized(tile.image_path, self.image_size)
        return q_img, t_img, idx
