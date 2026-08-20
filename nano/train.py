"""Nano: DINOv2-tiny (dinov2_vits14, ~21M backbone params) + SALAD, trained on
the WHOLE available query/tile pool (not astroloc/data/regions.py's 6-region
subset) with dynamic (paper-faithful, periodically reclustered) batching, on
reference tiles restricted to zoom level 9 only.

This is the efficiency-first sibling of astroloc/: same architecture family
and losses (L_pairs + L_MUM), same training loop shape, but sized and scoped
for a nanosatellite target instead of a demo/accuracy target -- see
nano/README.md for the actual measured numbers (params, index size, eval RAM)
once this has run, rather than trusting this docstring's intent.

pretrained=True is NOT available here: the official SALAD checkpoint
(astroloc/models/dinov2_salad.py::SALAD_CHECKPOINT_URL) was trained for
dinov2_vitb14 (768-dim tokens) and does not shape-match dinov2_vits14
(384-dim) -- DinoV2SaladModel raises if you try. The DINOv2 backbone itself
is still its own (self-supervised, LVD-142M) pretrained weights either way,
loaded by torch.hub inside third_party/salad's DINOv2 wrapper regardless of
this flag -- only the SALAD aggregator + reduction layer start from scratch.
"""

import argparse
import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from astroloc.data.pairing import build_positive_pairs
from astroloc.losses.mum import mum_loss
from astroloc.losses.pairwise import pairwise_loss
from astroloc.models.dinov2_salad import IMAGENET_MEAN, IMAGENET_STD, DinoV2SaladModel, DinoV2SaladRetriever
from astroloc.training.cluster import embed_tiles, kmeans_cluster, recluster
from astroloc.training.dataset import PairDataset
from astroloc.training.sampler import ClusterBatchSampler

from nano.data import build_query_set_all, build_reference_tiles_multi_zoom

BACKBONE_NAME = "dinov2_vits14"
# Matches the paper's own "AstroLoc-tiny" variant exactly: 512-dim (not the base
# model's 2048), confirmed from arXiv 2502.07003 text ("a linear layer to reduce
# feature dimensionality... AstroLoc-tiny... 27M parameters... memory required to
# store database features with AstroLoc-tiny is 12k x 512 x 4 x 4 = 98MB"). At
# 2048 this model would be ~40M params, not ~27M -- the reduction target is most
# of the difference (17.3M params at 2048 vs 4.3M at 512).
REDUCED_DIM = 512
TRAIN_DIR = "/mnt/sdc1/astroloc/reference_db/nano_train"
CACHE_DIR = os.path.join(TRAIN_DIR, "cache")
CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")


def build_training_data(
    database_dir: str,
    earthloc_queries_dir: str,
    gape_queries_dir: str,
    iou_threshold: float,
    num_clusters: int,
    device: str,
    rebuild: bool,
) -> dict:
    cache_path = os.path.join(CACHE_DIR, "pairs_cache.pkl")
    if not rebuild and os.path.exists(cache_path):
        print(f"Loading cached training pairs from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Building multi-zoom (9+10) reference tile set for training pairs (whole pool, minus eval regions)...")
    tiles = build_reference_tiles_multi_zoom(database_dir)

    print("Building query set (whole pool, minus eval regions)...")
    queries = build_query_set_all(earthloc_queries_dir, gape_queries_dir)

    print(f"Building positive pairs (IoU >= {iou_threshold})...")
    t0 = time.time()
    pairs = build_positive_pairs(queries, tiles, iou_threshold=iou_threshold)
    print(f"{len(pairs)}/{len(queries)} queries paired in {time.time() - t0:.0f}s")

    print("Embedding reference tiles (random-init SALAD, pretrained backbone only) for initial clustering...")
    pretrained_model = DinoV2SaladModel(pretrained=False, backbone_name=BACKBONE_NAME, reduced_dim=REDUCED_DIM)
    retriever = DinoV2SaladRetriever(pretrained_model, device=device)
    tile_embeddings = embed_tiles(retriever, tiles)
    _, tile_cluster_ids = kmeans_cluster(tile_embeddings, k=num_clusters)
    tile_id_to_cluster = {t.tile_id: int(c) for t, c in zip(tiles, tile_cluster_ids)}
    cluster_ids = [tile_id_to_cluster[tile.tile_id] for _, tile in pairs]
    del pretrained_model, retriever
    torch.cuda.empty_cache()

    data = {"pairs": pairs, "cluster_ids": cluster_ids, "num_clusters": num_clusters}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cached training pairs to {cache_path}")
    return data


def _checkpoint_payload(model, step: int, args) -> dict:
    return {
        "model": model.state_dict(),
        "step": step,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "backbone_name": BACKBONE_NAME,
        "reduced_dim": REDUCED_DIM,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--earthloc-queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--gape-queries-dir", default="/mnt/sdc1/astroloc/reference_db/astroloc_train/gape_queries")
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--num-clusters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--wandb-run-name", default="nano-dinov2s-dynamic")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--recluster-every-steps", type=int, default=5000)
    ap.add_argument("--max-pairs", type=int, default=None, help="debug: trim to this many cached pairs")
    args = ap.parse_args()
    checkpoint_dir = args.checkpoint_dir

    data = build_training_data(
        args.database_dir,
        args.earthloc_queries_dir,
        args.gape_queries_dir,
        args.iou_threshold,
        args.num_clusters,
        args.device,
        args.rebuild_cache,
    )
    pairs = data["pairs"]
    cached_cluster_ids = data["cluster_ids"]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
        cached_cluster_ids = cached_cluster_ids[: args.max_pairs]
    print(f"Training on {len(pairs)} pairs, {data['num_clusters']} clusters")

    query_cluster_ids = np.array(cached_cluster_ids, dtype=np.int64)
    tile_cluster_ids = np.array(cached_cluster_ids, dtype=np.int64)

    unique_tiles = list({tile.tile_id: tile for _, tile in pairs}.values())
    unique_queries = list({query.tile_id: query for query, _ in pairs}.values())

    dataset = PairDataset(pairs)
    batch_sampler = ClusterBatchSampler(query_cluster_ids.tolist(), batch_size=args.batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = DinoV2SaladModel(
        pretrained=False,
        backbone_name=BACKBONE_NAME,
        reduced_dim=REDUCED_DIM,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(args.device)
    model.train()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.model.parameters())
    print(f"backbone ({BACKBONE_NAME}) params: {n_backbone/1e6:.2f}M")
    print(f"trainable params: {n_trainable/1e6:.2f}M / total {n_total/1e6:.1f}M ({100*n_trainable/n_total:.1f}%)")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)

    def preprocess(batch_uint8: torch.Tensor) -> torch.Tensor:
        x = batch_uint8.to(args.device, non_blocking=True).float() / 255.0
        return (x - mean) / std

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    os.makedirs(checkpoint_dir, exist_ok=True)
    step = 0
    t_start = time.time()
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total steps planned")

    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for q_img, t_img, idx in loader:
            step_t0 = time.time()
            q = preprocess(q_img)
            t_ = preprocess(t_img)
            idx_np = idx.numpy()
            q_cids = torch.from_numpy(query_cluster_ids[idx_np]).to(args.device)
            t_cids = torch.from_numpy(tile_cluster_ids[idx_np]).to(args.device)

            optimizer.zero_grad(set_to_none=True)
            combined = torch.cat([q, t_], dim=0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = model(combined)
            embeddings = embeddings.float()
            n = q.shape[0]
            q_emb, t_emb = embeddings[:n], embeddings[n:]

            l_pairs = pairwise_loss(q_emb, t_emb)
            l_mum = mum_loss(q_emb, t_emb, q_cids, t_cids)
            loss = l_pairs + l_mum
            loss.backward()
            optimizer.step()

            step += 1
            step_time = time.time() - step_t0

            if args.recluster_every_steps and step % args.recluster_every_steps == 0 and step < total_steps:
                print(f"step {step}: reclustering...", flush=True)
                t_recluster0 = time.time()
                model.eval()
                retriever = DinoV2SaladRetriever(model, device=args.device)
                tile_id_to_cluster, query_id_to_cluster = recluster(
                    retriever, unique_tiles, unique_queries, k=args.num_clusters
                )
                model.train()
                tile_cluster_ids = np.array(
                    [tile_id_to_cluster[tile.tile_id] for _, tile in pairs], dtype=np.int64
                )
                query_cluster_ids = np.array(
                    [query_id_to_cluster[query.tile_id] for query, _ in pairs], dtype=np.int64
                )
                batch_sampler.update_cluster_ids(query_cluster_ids.tolist())
                print(f"  reclustering done in {time.time() - t_recluster0:.0f}s", flush=True)

            if step % args.log_every == 0:
                elapsed_min = (time.time() - t_start) / 60
                print(
                    f"step {step}/{total_steps} epoch {epoch} loss={loss.item():.4f} "
                    f"l_pairs={l_pairs.item():.4f} l_mum={l_mum.item():.4f} "
                    f"step_time={step_time:.3f}s elapsed={elapsed_min:.1f}min",
                    flush=True,
                )
                if use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "loss": loss.item(),
                            "l_pairs": l_pairs.item(),
                            "l_mum": l_mum.item(),
                            "epoch": epoch,
                            "step_time_s": step_time,
                        },
                        step=step,
                    )

            if step % args.checkpoint_every == 0:
                torch.save(_checkpoint_payload(model, step, args), os.path.join(checkpoint_dir, "latest.pt"))
                print(f"checkpoint saved at step {step}", flush=True)

            if (time.time() - t_start) > args.max_hours * 3600:
                print("Max wall-clock budget reached, stopping early.")
                stop = True
                break

    torch.save(_checkpoint_payload(model, step, args), os.path.join(checkpoint_dir, "final.pt"))
    torch.save(_checkpoint_payload(model, step, args), os.path.join(checkpoint_dir, "latest.pt"))
    print(f"Training complete at step {step}, saved final.pt")
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
