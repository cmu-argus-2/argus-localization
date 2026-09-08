"""Cluster-quadruplet batch sampler: the paper-faithful counterpart to
ClusterBatchSampler, for AstroLoc's L_MUM batch. Per the paper's own text
(arXiv:2502.07003 section 5.1, "Implementation Details", pulled directly via
WebFetch, not re-guessed): "batch size = 48 (48 pairs for pair loss, 48
quadruplets for MUM loss)" -- two SEPARATE batches per step, not one 48-batch
split into 24+24 (a wrong figure a prior session recorded, see repo memory
astroloc_target_architecture's correction). A quadruplet is 4 images from the
same cluster.

This sampler only emits flat lists of TILE indices (length
`num_quadruplets * 4`); it does not attach cluster labels to the batch. The
training loop recovers labels itself via `tile_cluster_ids[idx]` on the
indices a DataLoader batch actually returns (same safe pattern
PairDataset/ClusterBatchSampler already use -- see dataset.py's docstring for
why: cluster ids can change between reclustering events, and doing the lookup
in the main process after the batch comes back avoids ever needing to mutate
state inside a forked DataLoader worker). Any two indices that land in the
same cluster are valid MUM positives for each other regardless of which
quadruplet-draw put them in the batch -- the quadruplet is just the sampling
mechanism that guarantees each drawn cluster gets >=4 members represented,
not a grouping the loss itself needs to know about.

Cluster choice is weighted by query popularity (how many queries currently
map to each cluster) -- the paper's own stated rationale for cluster sampling
("clusters ... sampled according to how many queries are assigned to each
cluster") applies here exactly as it does for ClusterBatchSampler's pairs.
"""

import random
from collections import defaultdict

from torch.utils.data import Sampler


class QuadrupletBatchSampler(Sampler):
    def __init__(
        self,
        tile_cluster_ids: list[int],
        query_cluster_ids_for_weighting: list[int],
        num_quadruplets: int = 48,
        seed: int = 0,
    ):
        self.n_tiles = len(tile_cluster_ids)
        self.num_quadruplets = num_quadruplets
        self.rng = random.Random(seed)
        self.tile_cluster_ids = tile_cluster_ids
        self.query_cluster_ids_for_weighting = query_cluster_ids_for_weighting
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        by_cluster = defaultdict(list)
        for i, c in enumerate(self.tile_cluster_ids):
            by_cluster[c].append(i)
        self.by_cluster = by_cluster

        weight_counts: dict[int, int] = defaultdict(int)
        for c in self.query_cluster_ids_for_weighting:
            weight_counts[c] += 1
        # Only clusters with >=1 tile are drawable; fall back to uniform over
        # tile-clusters if query-side weighting doesn't overlap them at all.
        keys = [c for c in by_cluster if weight_counts.get(c, 0) > 0]
        if not keys:
            keys = list(by_cluster.keys())
            weights = [1] * len(keys)
        else:
            weights = [weight_counts[c] for c in keys]
        self.cluster_keys = keys
        self.cluster_weights = weights

    def update_cluster_ids(
        self, tile_cluster_ids: list[int], query_cluster_ids_for_weighting: list[int]
    ) -> None:
        assert len(tile_cluster_ids) == self.n_tiles
        self.tile_cluster_ids = tile_cluster_ids
        self.query_cluster_ids_for_weighting = query_cluster_ids_for_weighting
        self._rebuild_index()

    def __iter__(self):
        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(self.num_quadruplets):
                c = self.rng.choices(self.cluster_keys, weights=self.cluster_weights, k=1)[0]
                members = self.by_cluster[c]
                if len(members) >= 4:
                    picked = self.rng.sample(members, 4)
                else:
                    picked = [self.rng.choice(members) for _ in range(4)]
                batch.extend(picked)
            yield batch

    def __len__(self) -> int:
        denom = self.num_quadruplets * 4
        return max(1, self.n_tiles // denom)
