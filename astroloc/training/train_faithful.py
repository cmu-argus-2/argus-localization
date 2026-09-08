"""Paper-faithful AstroLoc training: a separate 48-pair (96-image) batch for
L_pairs and a separate 48-quadruplet (192-image) batch for L_MUM, two forward
passes per step -- matching arXiv:2502.07003 section 5.1 exactly ("batch
size = 48 (48 pairs for pair loss, 48 quadruplets for MUM loss)"), pulled
directly via WebFetch 2026-08-26, not this repo's earlier one-shared-batch
simplification (see astroloc/training/quadruplet_sampler.py's docstring and
repo memory astroloc_target_architecture's correction).

Loads the shared cache from build_full_data.py -- every run of this script
against the same cache file only differs in --use-lora / --dynamic-batching /
--recluster-every-steps, isolating exactly those two variables between the
"current best proven" (LoRA, static) and "wholly faithful" (full FT, dynamic)
runs agreed with the user.
"""

import argparse
import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from astroloc.losses.multi_similarity import multi_similarity_loss
from astroloc.losses.pairwise import pairwise_loss
from astroloc.models.dinov2_salad import IMAGENET_MEAN, IMAGENET_STD, DinoV2SaladModel, DinoV2SaladRetriever
from astroloc.training.cluster import recluster
from astroloc.training.dataset import PairDataset, TileDataset
from astroloc.training.quadruplet_sampler import QuadrupletBatchSampler
from astroloc.training.sampler import ClusterBatchSampler

DATA_CACHE = "/mnt/sdc1/astroloc/reference_db/astroloc_train/cache/full_pairs_cache.pkl"


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _checkpoint_payload(model, step: int, args) -> dict:
    return {
        "model": model.state_dict(),
        "step": step,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "faithful_batching": True,
        "dynamic_batching": args.dynamic_batching,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-cache", default=DATA_CACHE)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-quadruplets", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--wandb-run-name", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--dynamic-batching", action="store_true")
    ap.add_argument("--recluster-every-steps", type=int, default=0)
    args = ap.parse_args()

    with open(args.data_cache, "rb") as f:
        data = pickle.load(f)
    train_pairs = data["train_pairs"]
    val_pairs = data["val_pairs"]
    cached_cluster_ids = data["cluster_ids"]
    num_clusters = data["num_clusters"]
    print(f"Training on {len(train_pairs)} train pairs / {len(val_pairs)} val pairs, {num_clusters} clusters", flush=True)

    query_cluster_ids = np.array(cached_cluster_ids, dtype=np.int64)
    unique_tiles = list({tile.tile_id: tile for _, tile in train_pairs}.values())
    tile_id_to_idx = {t.tile_id: i for i, t in enumerate(unique_tiles)}
    tile_cluster_ids = np.zeros(len(unique_tiles), dtype=np.int64)
    for (_, tile), c in zip(train_pairs, cached_cluster_ids):
        tile_cluster_ids[tile_id_to_idx[tile.tile_id]] = c
    unique_queries = list({q.tile_id: q for q, _ in train_pairs}.values())
    print(f"{len(unique_tiles)} unique train tiles, {len(unique_queries)} unique train queries", flush=True)

    # --- L_pairs stream ---
    pair_dataset = PairDataset(train_pairs)
    if args.dynamic_batching:
        pair_sampler = ClusterBatchSampler(query_cluster_ids.tolist(), batch_size=args.batch_size)
        pair_loader = DataLoader(
            pair_dataset, batch_sampler=pair_sampler, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=args.num_workers > 0,
        )
    else:
        pair_sampler = None
        pair_loader = DataLoader(
            pair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0,
        )

    # --- L_MUM stream (quadruplets) ---
    tile_dataset = TileDataset(unique_tiles)
    quad_sampler = QuadrupletBatchSampler(
        tile_cluster_ids.tolist(), query_cluster_ids.tolist(), num_quadruplets=args.num_quadruplets,
    )
    quad_loader = DataLoader(
        tile_dataset, batch_sampler=quad_sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    # --- val (pairs-only loss signal) ---
    val_loader = None
    if len(val_pairs) >= args.batch_size:
        val_dataset = PairDataset(val_pairs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = DinoV2SaladModel(
        pretrained=True, use_lora=args.use_lora, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(args.device)
    model.train()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable/1e6:.2f}M / total {n_total/1e6:.1f}M ({100*n_trainable/n_total:.1f}%)", flush=True)
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

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    pair_iter = _infinite(pair_loader)
    quad_iter = _infinite(quad_loader)
    val_iter = _infinite(val_loader) if val_loader is not None else None

    t_start = time.time()
    for step in range(1, args.total_steps + 1):
        step_t0 = time.time()

        # Two separate backward() calls instead of backward() on (l_pairs +
        # l_mum) once -- mathematically identical (gradients accumulate
        # additively across backward() calls onto the same .grad buffers,
        # exactly matching grad-of-a-sum), but only ONE of the two batches'
        # activation graphs (96-image pairs vs 192-image quadruplets) is ever
        # held in memory at a time instead of both simultaneously. Needed
        # because --use-lora requires gradients through all 12 backbone
        # blocks (adapters touch every block) instead of full fine-tune's
        # last-4-only, so LoRA's peak activation memory is actually HIGHER
        # than full fine-tune's despite far fewer trainable params -- this
        # OOM'd at 22.43/23.49 GiB on the very first step before this fix.
        optimizer.zero_grad(set_to_none=True)

        q_img, t_img, _ = next(pair_iter)
        q = preprocess(q_img)
        t_ = preprocess(t_img)
        combined_pairs = torch.cat([q, t_], dim=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb_pairs = model(combined_pairs)
        emb_pairs = emb_pairs.float()
        n_q = q.shape[0]
        l_pairs = pairwise_loss(emb_pairs[:n_q], emb_pairs[n_q:])
        l_pairs.backward()

        quad_img, quad_idx = next(quad_iter)
        quad_idx_np = quad_idx.numpy()
        quad_labels = torch.from_numpy(tile_cluster_ids[quad_idx_np]).to(args.device)
        quad_x = preprocess(quad_img)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb_quad = model(quad_x)
        emb_quad = emb_quad.float()
        l_mum = multi_similarity_loss(emb_quad, quad_labels, alpha=1.0, beta=50.0)
        l_mum.backward()

        optimizer.step()
        loss = l_pairs.detach() + l_mum.detach()
        step_time = time.time() - step_t0

        if (
            args.dynamic_batching
            and args.recluster_every_steps
            and step % args.recluster_every_steps == 0
            and step < args.total_steps
        ):
            print(f"step {step}: reclustering...", flush=True)
            t_r0 = time.time()
            model.eval()
            retriever = DinoV2SaladRetriever(model, device=args.device)
            tile_id_to_cluster, query_id_to_cluster = recluster(retriever, unique_tiles, unique_queries, k=num_clusters)
            model.train()
            tile_cluster_ids = np.array([tile_id_to_cluster[t.tile_id] for t in unique_tiles], dtype=np.int64)
            query_cluster_ids = np.array([query_id_to_cluster[q.tile_id] for q, _ in train_pairs], dtype=np.int64)
            pair_sampler.update_cluster_ids(query_cluster_ids.tolist())
            quad_sampler.update_cluster_ids(tile_cluster_ids.tolist(), query_cluster_ids.tolist())
            print(f"  reclustering done in {time.time() - t_r0:.0f}s", flush=True)

        if step % args.log_every == 0:
            elapsed_min = (time.time() - t_start) / 60
            print(
                f"step {step}/{args.total_steps} loss={loss.item():.4f} l_pairs={l_pairs.item():.4f} "
                f"l_mum={l_mum.item():.4f} step_time={step_time:.3f}s elapsed={elapsed_min:.1f}min",
                flush=True,
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {"loss": loss.item(), "l_pairs": l_pairs.item(), "l_mum": l_mum.item(), "step_time_s": step_time},
                    step=step,
                )

        if val_iter is not None and step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                vq_img, vt_img, _ = next(val_iter)
                vq = preprocess(vq_img)
                vt = preprocess(vt_img)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    emb_v = model(torch.cat([vq, vt], dim=0))
                emb_v = emb_v.float()
                nv = vq.shape[0]
                val_l_pairs = pairwise_loss(emb_v[:nv], emb_v[nv:])
            model.train()
            print(f"  step {step}: val_l_pairs={val_l_pairs.item():.4f}", flush=True)
            if use_wandb:
                import wandb

                wandb.log({"val_l_pairs": val_l_pairs.item()}, step=step)

        if step % args.checkpoint_every == 0:
            torch.save(_checkpoint_payload(model, step, args), os.path.join(args.checkpoint_dir, "latest.pt"))

    torch.save(_checkpoint_payload(model, args.total_steps, args), os.path.join(args.checkpoint_dir, "final.pt"))
    torch.save(_checkpoint_payload(model, args.total_steps, args), os.path.join(args.checkpoint_dir, "latest.pt"))
    print(f"Training complete at step {args.total_steps}, saved final.pt", flush=True)
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
