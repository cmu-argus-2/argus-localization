"""v0 default retriever: wraps the released EarthLoc model (DINOv2-base + SALAD).

This validates plumbing and gets a baseline recall number fast. It is NOT the
deployment retriever, see docs/argus_localization_design.md section 6 and
docs/argus_localization_spec.md section 5. The deployment target is
SmallRetriever in retrievers/small_retriever.py.

This is the only module allowed to import EarthLoc code. It imports only
apl_models.apl_model.APLModel (the network definition) from the vendored
reference copy in third_party/EarthLoc. Everything else, including EarthLoc's
own datasets/utils.py, is reimplemented here so nothing else from EarthLoc
leaks into the rest of the pipeline.

Preprocessing mirrors datasets/utils.py::load_image (resize, to-tensor scaled
to [0,1], ImageNet normalize) but runs as one batched operation on the GPU
instead of PIL Resize + ToTensor + Normalize per image on the CPU. Measured
on this machine: the per-image CPU path tops out around 130 img/s and leaves
the GPU over 80% idle (a batched GPU forward pass alone does over 1000 img/s);
batching the resize and normalize on GPU instead measured about 285 img/s,
with resulting descriptors at cosine similarity > 0.99 against the per-image
CPU path (small numerical differences from bilinear-with-antialias resize
implementation, not a behavior change).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from core.interfaces import Retriever

_EARTHLOC_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "EarthLoc"
)


def _load_apl_model_class():
    # apl_model.py does "from apl_models import mixvpr", so apl_models must be
    # importable as a top level package while this import runs. Add the
    # EarthLoc root to sys.path only for the duration of the import, then
    # remove it, so no other EarthLoc module stays reachable afterward.
    added = _EARTHLOC_ROOT not in sys.path
    if added:
        sys.path.insert(0, _EARTHLOC_ROOT)
    try:
        from apl_models.apl_model import APLModel
    finally:
        if added:
            sys.path.remove(_EARTHLOC_ROOT)
    return APLModel


def _load_state_dict(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and not all(torch.is_tensor(v) for v in checkpoint.values()):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]
        raise ValueError(
            f"checkpoint at {checkpoint_path} is a dict but has none of the expected "
            "state dict keys (model_state_dict, state_dict, model)"
        )
    return checkpoint


class EarthLocRetriever(Retriever):
    descriptor_dim: int

    def __init__(self, checkpoint_path: str, device: str = "cuda", image_size: int = 320):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = image_size
        self.descriptor_dim = 4096  # fixed by the released checkpoint's architecture

        apl_model_cls = _load_apl_model_class()
        self.model = apl_model_cls(image_size=image_size, desc_dim=self.descriptor_dim)
        self.model.load_state_dict(_load_state_dict(checkpoint_path, device))
        self.model = self.model.to(device).eval()

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if len({img.shape for img in images}) > 1:
            # Mixed shapes in one call are not the common case (see module docstring:
            # a given call is always all db tiles or all queries, which share a fixed
            # pixel size), but resize to a common size on CPU so the batch can still
            # be stacked into one tensor.
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
        batch = (batch - self._mean) / self._std

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    descriptors = self.model(batch)
            else:
                descriptors = self.model(batch)
        return descriptors.float().cpu().numpy().astype(np.float32)
