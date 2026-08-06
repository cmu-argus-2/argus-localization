"""Minimal from-scratch LoRA (Hu et al. 2021) for the DINOv2 backbone.

Hand-rolled instead of depending on `peft`: the vendored DINOv2 class
(third_party/salad/models/backbones/dinov2.py) isn't a HF model, and a ~40
line implementation is easy to verify directly (see the base-vs-lora-output
equality check in astroloc/training/train_lora_smoketest.py-style checks run
before the real training job). Wraps every nn.Linear inside the given module
(qkv/proj/mlp fc1/fc2 for a standard ViT block, whatever the exact attribute
names) with a frozen base + a trainable low-rank A/B update, B initialized to
zero so the wrapped model's output is bit-identical to the un-wrapped one at
the start of training.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # lora_B stays 0-init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def apply_lora_to_linears(module: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    """Recursively replaces every nn.Linear child with a LoRALinear wrapping
    it. Returns the number of Linear layers wrapped.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += apply_lora_to_linears(child, r=r, alpha=alpha, dropout=dropout)
    return count
