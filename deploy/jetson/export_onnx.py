"""Export a trained DinoV2+SALAD retriever (astroloc/ full-size or nano/) to ONNX,
then verify the exported graph produces the same descriptors as PyTorch.

Run from the argus-localization repo root, on the workstation (needs the
checkpoint, third_party/salad, and a torch.hub cache for DINOv2):

    # nano (dinov2_vits14, 512-dim)
    python export_onnx.py \
        --checkpoint /mnt/sdc1/astroloc/reference_db/nano_train/checkpoints/final.pt \
        --output nano.onnx

    # full-size (dinov2_vitb14, 2048-dim)
    python export_onnx.py \
        --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints/final.pt \
        --output astroloc.onnx

Architecture is read from the checkpoint by DinoV2SaladRetriever.from_checkpoint
(backbone_name, reduced_dim, use_lora, lora_r, lora_alpha), so the same script
handles both model sizes with no flags.

Exports the model only -- NOT the resize/normalize preprocessing, which stays in
Python (DinoV2SaladRetriever._preprocess). Whatever runs this graph on the Orin
must feed an already-resized, ImageNet-normalized (B,3,224,224) float32 tensor.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Resolve the argus-localization repo root. Defaults to the current working
# directory (run this from the repo root); override with ARGUS_LOCALIZATION_ROOT
# if you keep this script elsewhere.
_REPO_ROOT = os.environ.get("ARGUS_LOCALIZATION_ROOT", os.getcwd())
if not os.path.isdir(os.path.join(_REPO_ROOT, "astroloc")):
    sys.exit(
        f"ERROR: no astroloc/ package under {_REPO_ROOT}\n"
        "  Run this from the argus-localization repo root, or set\n"
        "  ARGUS_LOCALIZATION_ROOT=/path/to/argus-localization"
    )
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from astroloc.models.dinov2_salad import IMAGE_SIZE, DinoV2SaladRetriever


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="path to final.pt")
    ap.add_argument("--output", required=True, help="path to write the .onnx file")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu. Export on cpu if GPU memory is tight; the "
                         "resulting graph is device-independent.")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help=f"square input size (default {IMAGE_SIZE}, divisible by the "
                         "ViT patch size 14)")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset (default 18). torch 2.14 cannot emit below 18 -- "
                         "asking for less silently yields 18 plus a noisy (harmless) "
                         "version-converter traceback, because DINOv2's Resize op has "
                         "no downconversion adapter. Verified by direct test.")
    ap.add_argument("--batch", type=int, default=2,
                    help="dummy batch size used for tracing and parity (>1 so a "
                         "batch-collapsing bug can't hide)")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="max allowed abs difference between torch and onnxruntime "
                         "descriptors")
    ap.add_argument("--skip-verify", action="store_true",
                    help="export only; skip the onnxruntime parity check")
    ap.add_argument("--static-batch", action="store_true",
                    help="fix the batch dimension instead of making it dynamic. "
                         "Required for TensorRT: SALAD's Sinkhorn uses bare .squeeze() "
                         "calls with no axis argument, and TRT cannot infer which axes "
                         "to drop from a dynamic shape (ModelImporter error at "
                         "/aggregator/Squeeze). The engine is then locked to --batch.")
    return ap.parse_args()


def load_retriever(checkpoint_path, device="cuda"):
    """DinoV2SaladRetriever.from_checkpoint, plus a fix for LoRA checkpoints saved
    before commit cc75c2b (Sept 8). That commit wraps each DINOv2 block in
    _CheckpointedBlock, which adds a ".block." level to the weight names, so older
    checkpoints (e.g. nano_train/checkpoints_v2, Aug 20) no longer load. The weights
    are unchanged; only the names are remapped. Loading stays strict."""
    import re
    import torch
    from astroloc.models.dinov2_salad import REDUCED_DIM, DinoV2SaladModel, DinoV2SaladRetriever

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DinoV2SaladModel(
        pretrained=False,
        use_lora=state.get("use_lora", False),
        lora_r=state.get("lora_r", 8),
        lora_alpha=state.get("lora_alpha", 16),
        backbone_name=state.get("backbone_name", "dinov2_vitb14"),
        reduced_dim=state.get("reduced_dim", REDUCED_DIM),
    )
    weights = state["model"] if "model" in state else state
    expected = set(model.state_dict())
    if set(weights) != expected:
        remapped = {re.sub(r"^(backbone\.model\.blocks\.\d+\.)(?!block\.)", r"\1block.", k): v
                    for k, v in weights.items()}
        if set(remapped) == expected:
            print("  note: remapped pre-Sept-8 checkpoint weight names (added '.block.')")
            weights = remapped
    model.load_state_dict(weights)
    return DinoV2SaladRetriever(model, device=device)


def main() -> int:
    args = parse_args()

    if args.image_size % 14 != 0:
        print(f"ERROR: --image-size {args.image_size} is not divisible by 14 "
              f"(DINOv2 patch size). Use {round(args.image_size / 14) * 14}.")
        return 1

    print(f"Loading checkpoint: {args.checkpoint}")
    retriever = load_retriever(args.checkpoint, device=args.device)
    model = retriever.model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  backbone      : {model.backbone_name if hasattr(model, 'backbone_name') else 'see checkpoint'}")
    print(f"  descriptor dim: {retriever.descriptor_dim}")
    print(f"  parameters    : {n_params / 1e6:.1f}M")

    # eval() already disables _CheckpointedBlock's activation checkpointing and
    # LoRA dropout; no_grad keeps the trace free of autograd bookkeeping.
    dummy = torch.randn(args.batch, 3, args.image_size, args.image_size,
                        device=args.device)

    mode = f"static batch={args.batch}" if args.static_batch else "dynamic batch"
    print(f"Exporting -> {args.output} (opset {args.opset}, {mode})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            args.output,
            input_names=["images"],
            output_names=["descriptors"],
            # Batch is dynamic so one engine serves any batch size; H/W stay fixed
            # because DINOv2's positional embeddings are interpolated per-resolution
            # and a dynamic spatial axis would bake in whatever it traced.
            dynamic_axes=(None if args.static_batch else
                          {"images": {0: "batch"}, "descriptors": {0: "batch"}}),
            opset_version=args.opset,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  wrote {size_mb:.1f} MB")

    # Weights are stored in a sidecar .data file. Both must travel to the Jetson.
    sidecar = args.output + ".data"
    if os.path.exists(sidecar):
        print(f"  wrote {os.path.getsize(sidecar)/1e6:.1f} MB  {os.path.basename(sidecar)}"
              f"  <-- external weights, copy alongside the .onnx")

    # torch may silently fall back to a higher opset than requested; surface it.
    try:
        import onnx as _onnx
        actual = _onnx.load(args.output, load_external_data=False).opset_import[0].version
        if actual != args.opset:
            print(f"  NOTE: requested opset {args.opset}, file is opset {actual} "
                  f"(exporter fell back). Check your TensorRT supports {actual}.")
        else:
            print(f"  opset {actual} as requested")
    except Exception:
        pass

    if args.skip_verify:
        print("Skipping verification (--skip-verify).")
        return 0

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("\nWARNING: onnx/onnxruntime not installed, cannot verify.")
        print("  pip install onnx onnxruntime    (or onnxruntime-gpu)")
        print("  Then re-run, or run with --skip-verify to silence this.")
        return 0

    print("Checking graph structure")
    onnx.checker.check_model(onnx.load(args.output))
    print("  graph is well-formed")

    print("Comparing descriptors: PyTorch vs onnxruntime")
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["descriptors"], {"images": dummy.cpu().numpy()})[0]

    if torch_out.shape != onnx_out.shape:
        print(f"  FAIL: shape mismatch, torch {torch_out.shape} vs onnx {onnx_out.shape}")
        return 1

    max_diff = float(np.abs(torch_out - onnx_out).max())
    # Descriptors are L2-normalized, so cosine similarity is the metric that
    # actually matters downstream -- retrieval ranks by it.
    cos = float((torch_out * onnx_out).sum(axis=1).min())

    print(f"  shape            : {torch_out.shape}")
    print(f"  max abs diff     : {max_diff:.2e}  (tolerance {args.tolerance:.0e})")
    print(f"  min cosine sim   : {cos:.6f}  (want > 0.9999)")

    if max_diff > args.tolerance:
        print("\nFAIL: descriptors diverge beyond tolerance.")
        print("  The export is not numerically faithful -- do not use it for retrieval.")
        return 1

    print("\nPASS: ONNX export matches PyTorch.")
    print("Next: convert to a TensorRT engine on the Orin (FP32 -- FP16 drifts, see RUNBOOK), e.g.")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={os.path.basename(args.output)} "
          f"--saveEngine={os.path.splitext(os.path.basename(args.output))[0]}_fp32.trt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
