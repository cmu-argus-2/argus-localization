"""Plumbing check: reproduce EarthLoc retrieval recall using EarthLoc's own
model, dataset, and test harness directly, not the Argus pipeline. This only
proves the released checkpoint and the rsynced data load and run correctly.
See docs/argus_localization_spec.md section 9, Phase 0.

This is the one script in the repo allowed to import EarthLoc code freely
(apl_models, datasets, test, visualizations), because it is a reproduction
of EarthLoc's own eval.py, not part of the Argus pipeline. The pipeline
itself only ever imports apl_models.apl_model.APLModel, see
retrievers/earthloc_retriever.py.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EARTHLOC_ROOT = os.path.join(_REPO_ROOT, "third_party", "EarthLoc")
sys.path.insert(0, _EARTHLOC_ROOT)

import torch
import yaml

import test as earthloc_test
from apl_models.apl_model import APLModel
from datasets.test_dataset import TestDataset


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce EarthLoc recall on one region with EarthLoc's own code."
    )
    parser.add_argument("--user-config", default=os.path.join(_REPO_ROOT, "user_config.yaml"))
    parser.add_argument("--region-name", default="Alps")
    parser.add_argument("--center-lat", type=float, default=45)
    parser.add_argument("--center-lon", type=float, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)
    dataset_path = Path(user_config["data_root"])

    logging.info(f"Loading EarthLoc model from {user_config['earthloc_checkpoint']}")
    model = APLModel()
    state_dict = torch.load(user_config["earthloc_checkpoint"], map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval()

    logging.info("Listing 2021 reference tiles...")
    db_paths_from_2021 = list((dataset_path / "database").glob("*/*/*.jpg"))
    logging.info(f"{len(db_paths_from_2021)} reference tiles from 2021")

    test_dataset = TestDataset(
        dataset_path=dataset_path,
        dataset_name=args.region_name,
        db_paths=db_paths_from_2021,
        image_size=model.image_size,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
    )
    logging.info(f"Built {test_dataset}")

    recalls, recalls_str = earthloc_test.test(test_dataset, model, device=args.device)
    logging.info(f"Recalls on {args.region_name}: {recalls_str}")
    print(f"\n=== EarthLoc reproduction, region={args.region_name} ===")
    print(recalls_str)


if __name__ == "__main__":
    main()
