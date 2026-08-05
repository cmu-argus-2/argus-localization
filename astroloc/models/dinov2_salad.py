"""AstroLoc-style retriever: DINOv2-base backbone + SALAD aggregation (Sinkhorn
optimal-transport aggregation), an 8448-dim descriptor further reduced to
2048-dim by a linear layer, per the AstroLoc paper (arXiv:2502.07003) section
on model architecture. See docs note in the repo memory (astroloc_target_architecture)
for the exact numbers this mirrors.

This is the only module allowed to import the vendored `third_party/salad`
code (github.com/serizba/salad, the official SALAD release), mirroring the
existing convention in retrievers/earthloc_retriever.py for third_party/EarthLoc:
only the backbone (`models.backbones.dinov2.DINOv2`) and aggregator
(`models.aggregators.salad.SALAD`) classes are imported, nothing else (their
training script, PyTorch Lightning wrapper, and loss/miner utilities are not
used -- this repo has its own training loop and losses, see astroloc/losses/).

Weight loading: the official pretrained checkpoint (GSV-Cities place
recognition, downloaded from the salad GitHub release) is a flat state_dict
with only `backbone.*` and `aggregator.*` keys, because upstream's own
VPRModel also only has those two stateful submodules. `_DinoSaladBackboneAgg`
below is named the same way so `load_state_dict` lines up exactly with no
key remapping.
"""

import os
import sys

# Must be set before any torch.hub.load call (DINOv2 backbone construction pulls
# ~330MB of weights + repo code). Home has only ~27GB free on this machine, so
# torch.hub's default ~/.cache/torch is not safe to use for anything nontrivial.
os.environ.setdefault(
    "TORCH_HOME", "/mnt/sdc1/astroloc/reference_db/astroloc_train/cache/torch"
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import Retriever

_SALAD_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
    "salad",
)

SALAD_CHECKPOINT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"

# Matches hubconf.py::dinov2_salad's own defaults for backbone='dinov2_vitb14'.
NUM_CHANNELS = 768  # DINOv2 ViT-B/14 token dim
NUM_CLUSTERS = 64
CLUSTER_DIM = 128
TOKEN_DIM = 256
SALAD_DESCRIPTOR_DIM = TOKEN_DIM + NUM_CLUSTERS * CLUSTER_DIM  # 256 + 8192 = 8448, matches the paper
REDUCED_DIM = 2048  # AstroLoc paper's post-SALAD linear reduction

IMAGE_SIZE = 224  # divisible by 14 (patch size); matches SALAD's own GSV-Cities training resolution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_salad_classes():
    added = _SALAD_ROOT not in sys.path
    if added:
        sys.path.insert(0, _SALAD_ROOT)
    try:
        from models.aggregators.salad import SALAD
        from models.backbones.dinov2 import DINOv2
    finally:
        if added:
            sys.path.remove(_SALAD_ROOT)
    return DINOv2, SALAD


class DinoV2SaladModel(nn.Module):
    """Trainable network: DINOv2 backbone + SALAD aggregator + linear reduction.

    Attribute names `backbone`/`aggregator` are load-bearing: they must match
    the pretrained checkpoint's key prefixes exactly (see module docstring).
    """

    def __init__(self, num_trainable_blocks: int = 4, pretrained: bool = True):
        super().__init__()
        DINOv2, SALAD = _load_salad_classes()

        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=num_trainable_blocks,
            norm_layer=True,
            return_token=True,
        )
        self.aggregator = SALAD(
            num_channels=NUM_CHANNELS,
            num_clusters=NUM_CLUSTERS,
            cluster_dim=CLUSTER_DIM,
            token_dim=TOKEN_DIM,
        )
        self.reduction = nn.Linear(SALAD_DESCRIPTOR_DIM, REDUCED_DIM)

        if pretrained:
            self._load_pretrained_salad()
        self._freeze_backbone_stem()

    def _load_pretrained_salad(self) -> None:
        state_dict = torch.hub.load_state_dict_from_url(
            SALAD_CHECKPOINT_URL, map_location="cpu"
        )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # `reduction.*` is expected to be missing (it doesn't exist upstream, it's
        # AstroLoc's own addition); anything else missing/unexpected is a real bug.
        bad_missing = [k for k in missing if not k.startswith("reduction.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"unexpected state_dict mismatch loading pretrained SALAD: "
                f"missing={bad_missing}, unexpected={unexpected}"
            )

    def _freeze_backbone_stem(self) -> None:
        # DINOv2's own forward() already runs frozen blocks under torch.no_grad(),
        # but their params still default to requires_grad=True, which just wastes
        # optimizer memory/state for zero-gradient tensors. Freeze explicitly so
        # trainable_parameters() below (and the optimizer) only sees real ones.
        model = self.backbone.model
        for p in model.patch_embed.parameters():
            p.requires_grad_(False)
        model.cls_token.requires_grad_(False)
        model.pos_embed.requires_grad_(False)
        if model.mask_token is not None:
            model.mask_token.requires_grad_(False)
        n_trainable = self.backbone.num_trainable_blocks
        for blk in model.blocks[:-n_trainable]:
            for p in blk.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W) already resized/normalized. Returns (B,REDUCED_DIM) L2-normed."""
        f, t = self.backbone(images)
        descriptor = self.aggregator((f, t))  # (B, SALAD_DESCRIPTOR_DIM), already L2-normed
        descriptor = self.reduction(descriptor)
        return F.normalize(descriptor, p=2, dim=-1)


class DinoV2SaladRetriever(Retriever):
    """Retriever-protocol adapter around a (trained) DinoV2SaladModel, for use
    with database/reference_database.py, index/faiss_index.py, etc, exactly
    like retrievers/earthloc_retriever.py::EarthLocRetriever.
    """

    descriptor_dim: int = REDUCED_DIM

    def __init__(self, model: DinoV2SaladModel, device: str = "cuda", image_size: int = IMAGE_SIZE):
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = image_size
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda") -> "DinoV2SaladRetriever":
        model = DinoV2SaladModel(pretrained=False)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        return cls(model, device=device)

    def _preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        if len({img.shape for img in images}) > 1:
            import cv2

            images = [
                cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                for img in images
            ]
        batch = torch.from_numpy(np.stack(images)).to(self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        if batch.shape[-2:] != (self.image_size, self.image_size):
            batch = F.interpolate(
                batch, size=(self.image_size, self.image_size), mode="bilinear",
                align_corners=False, antialias=True,
            )
        return (batch - self._mean) / self._std

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        batch = self._preprocess(images)
        with torch.no_grad():
            descriptors = self.model(batch)
        return descriptors.float().cpu().numpy().astype(np.float32)
