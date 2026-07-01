"""Phase 2 target: small distilled retriever for on-board deployment.

timm EfficientNet-Lite or MobileNetV3 backbone + GeM pooling + L2 norm,
descriptor_dim 512, distilled from the EarthLoc teacher (relational,
pairwise-similarity distillation, not raw embedding copy) then domain
fine-tuned on Sentinel-2 tiles. See docs/argus_localization_design.md
section 6 and docs/argus_localization_spec.md section 5.

Not implemented in v0. Training happens in training/train_retriever.py,
which stays a stub until Phase 2.
"""

import numpy as np

from core.interfaces import Retriever


class SmallRetriever(Retriever):
    descriptor_dim: int = 512

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        raise NotImplementedError("Phase 2: backbone + GeM head, load distilled weights.")

    def embed(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError
