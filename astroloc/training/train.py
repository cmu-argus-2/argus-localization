"""Fine-tune DINOv2+SALAD (initialized from the official GSV-Cities pretrained
checkpoint) on GAPE mlcoord queries paired with the existing Sentinel-2-derived
reference tile mirror, using AstroLoc's two losses (L_pairs + L_MUM).

Simplifications vs. the AstroLoc paper (demo-scale, documented honestly, see
astroloc/README.md): clusters are computed once up front from the pretrained
model's embeddings (not recomputed periodically during training), and batches
are uniformly shuffled positive pairs rather than the paper's separate
pairs/quadruplet batch composition -- both losses are computed on the same
batch each step instead.
"""

import argparse
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.utils.data import DataLoader

from astroloc.data.pairing import build_positive_pairs
from astroloc.data.query_set import build_query_set
from astroloc.data.reference_tiles import build_reference_tiles
from astroloc.losses.mum import mum_loss
from astroloc.losses.pairwise import pairwise_loss
from astroloc.models.dinov2_salad import IMAGENET_MEAN, IMAGENET_STD, DinoV2SaladModel, DinoV2SaladRetriever
from astroloc.training.cluster import embed_tiles, kmeans_cluster
from astroloc.training.dataset import PairDataset

TRAIN_DIR = "/mnt/sdc1/astroloc/reference_db/astroloc_train"
CACHE_DIR = os.path.join(TRAIN_DIR, "cache")
CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")


def build_training_data(
    database_dir: str,
    earthloc_queries_dir: str,
    gape_queries_dir: str,
    tile_cap: int,
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

    print("Building reference tile set...")
    tiles = build_reference_tiles(database_dir, per_region_cap=tile_cap)

    print("Building query set...")
    queries = build_query_set(earthloc_queries_dir, gape_queries_dir)
    print(f"{len(queries)} candidate queries in training regions")

    print(f"Building positive pairs (IoU >= {iou_threshold})...")
    t0 = time.time()
    pairs = build_positive_pairs(queries, tiles, iou_threshold=iou_threshold)
    print(f"{len(pairs)}/{len(queries)} queries paired in {time.time() - t0:.0f}s")

    print("Embedding reference tiles with the pretrained checkpoint for initial clustering...")
    pretrained_model = DinoV2SaladModel(pretrained=True)
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
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--earthloc-queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument(
        "--gape-queries-dir", default=os.path.join(TRAIN_DIR, "gape_queries")
    )
    ap.add_argument("--tile-cap", type=int, default=10000)
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--num-clusters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-hours", type=float, default=13.0)
    ap.add_argument("--dry-run-steps", type=int, default=0)
    ap.add_argument("--use-lora", action="store_true", help="LoRA-adapt the whole frozen backbone instead of unfreezing the last few blocks")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    args = ap.parse_args()
    checkpoint_dir = args.checkpoint_dir

    data = build_training_data(
        args.database_dir,
        args.earthloc_queries_dir,
        args.gape_queries_dir,
        args.tile_cap,
        args.iou_threshold,
        args.num_clusters,
        args.device,
        args.rebuild_cache,
    )
    pairs, cluster_ids = data["pairs"], data["cluster_ids"]
    print(f"Training on {len(pairs)} pairs, {data['num_clusters']} clusters")

    dataset = PairDataset(pairs, cluster_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = DinoV2SaladModel(
        pretrained=True,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(args.device)
    model.train()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
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
        for q_img, t_img, cids in loader:
            step_t0 = time.time()
            q = preprocess(q_img)
            t_ = preprocess(t_img)
            cids = cids.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            combined = torch.cat([q, t_], dim=0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = model(combined)
            embeddings = embeddings.float()
            n = q.shape[0]
            q_emb, t_emb = embeddings[:n], embeddings[n:]

            l_pairs = pairwise_loss(q_emb, t_emb)
            l_mum = mum_loss(q_emb, t_emb, cids)
            loss = l_pairs + l_mum
            loss.backward()
            optimizer.step()

            step += 1
            step_time = time.time() - step_t0
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

            if args.dry_run_steps and step >= args.dry_run_steps:
                print("dry run complete")
                return

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
