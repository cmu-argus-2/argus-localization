"""Cluster-weighted batch sampler, replacing plain random shuffling with
something closer to AstroLoc's own batch composition: each batch is half
uniformly-random pairs, half drawn from a single cluster chosen weighted by
how many queries currently land in it ("clusters ... sampled according to
how many queries are assigned to each cluster").

Not a literal reproduction of the paper's separate 24-pairs/24-quadruplets
minibatches (see astroloc/README.md) -- this still yields ONE batch of pair
indices per step, just with the second half biased toward one popular
cluster instead of being fully random. `update_cluster_ids` lets the trainer
refresh which cluster each pair's query belongs to as reclustering happens
(astroloc/training/cluster.py::recluster), without rebuilding the sampler.
"""

import random
from collections import defaultdict

from torch.utils.data import Sampler


class ClusterBatchSampler(Sampler):
    def __init__(self, query_cluster_ids: list[int], batch_size: int = 48, seed: int = 0):
        self.n = len(query_cluster_ids)
        self.batch_size = batch_size
        self.n_random = batch_size // 2
        self.n_cluster = batch_size - self.n_random
        self.rng = random.Random(seed)
        self.query_cluster_ids = query_cluster_ids
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        by_cluster = defaultdict(list)
        for i, c in enumerate(self.query_cluster_ids):
            by_cluster[c].append(i)
        self.by_cluster = by_cluster
        self.cluster_keys = list(by_cluster.keys())
        self.cluster_weights = [len(v) for v in by_cluster.values()]  # query-popularity weighting

    def update_cluster_ids(self, query_cluster_ids: list[int]) -> None:
        assert len(query_cluster_ids) == self.n
        self.query_cluster_ids = query_cluster_ids
        self._rebuild_index()

    def __iter__(self):
        all_idx = list(range(self.n))
        for _ in range(len(self)):
            batch = self.rng.sample(all_idx, self.n_random)
            cluster_part: list[int] = []
            # Reads self.by_cluster/cluster_keys/cluster_weights fresh each
            # time, so a mid-epoch update_cluster_ids() call takes effect on
            # the very next batch of this same generator (no restart needed).
            while len(cluster_part) < self.n_cluster:
                c = self.rng.choices(self.cluster_keys, weights=self.cluster_weights, k=1)[0]
                members = self.by_cluster[c]
                take = min(len(members), self.n_cluster - len(cluster_part))
                cluster_part.extend(self.rng.sample(members, take))
            batch.extend(cluster_part)
            yield batch

    def __len__(self) -> int:
        return self.n // self.batch_size
