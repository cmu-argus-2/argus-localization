"""Phase 2 stub: distillation training for SmallRetriever.

Not run in v0 (docs/argus_localization_spec.md section 0 and 10 are explicit
that v0 does no training). Sketch only, so Phase 2 has a starting point.

Planned loss: relational distillation, match the teacher's (EarthLoc
DINOv2-base + SALAD) pairwise similarity structure within a batch rather than
copying raw embeddings, since recall@k depends on neighbor relationships, not
absolute vectors. Add a domain-specific fine-tune on Sentinel-2 tiles after
distillation. Augmentations: rotation, photometric jitter, multi-temporal
pairs (see docs/argus_localization_design.md section 6).
"""


def relational_distillation_loss(student_embeddings, teacher_embeddings):
    """Match pairwise cosine-similarity structure between student and teacher
    embeddings within a batch. Not implemented, Phase 2.
    """
    raise NotImplementedError


def train(config_path: str):
    raise NotImplementedError("Phase 2: distill SmallRetriever from the EarthLoc teacher.")


if __name__ == "__main__":
    raise SystemExit("train_retriever.py is a Phase 2 stub, not runnable in v0.")
