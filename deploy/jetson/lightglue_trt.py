"""LightGlue (SIFT weights) as a fixed-size network for ONNX / TensorRT.

LightGlue normally runs as many small PyTorch operations driven from Python (~850 GPU
kernel launches per tile pair), and on the Jetson the CPU launching them is the
bottleneck. This file rebuilds the same network, with the same weights and the same
layers, in a form TensorRT can compile into one engine:

- Fixed sizes: both images are padded to K points (default 1024). Padded points are
  masked out of every attention and softmax, so they can't change the result.
- All 9 layers always run. LightGlue's early stopping and point pruning depend on
  the data, so they can't be compiled. The PyTorch reference to compare against is
  LightGlue with depth_confidence = width_confidence = -1.
- The engine returns the log-assignment scores for the K x K point pairs. Turning
  them into matches (mutual best + threshold, as LightGlue's filter_matches does)
  happens outside, on the valid block only.

Export and check against PyTorch (CPU is enough), from the argus-localization repo root:
    python deploy/jetson/lightglue_trt.py export --out ~/lightglue_sift_k1024.onnx
Build the engine on the Jetson:
    /usr/src/tensorrt/bin/trtexec --onnx=lightglue_sift_k1024.onnx --saveEngine=lightglue_sift_k1024_fp16.trt --fp16
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

INPUTS = ("kpts0", "kpts1", "desc0", "desc1", "valid0", "valid1")


class LightGlueStatic(torch.nn.Module):
    """Same weights and maths as lightglue.LightGlue, all layers, padding masked out.
    Every reshape uses fixed numbers (batch 1, k points) so the exported graph has no
    shape arithmetic for ONNX/TensorRT to resolve."""

    def __init__(self, lg, k=1024, b=1, layers=None):
        super().__init__()
        self.lg, self.k, self.b = lg, k, b
        self.n_layers = layers or len(lg.transformers)  # stop after this many layers, using that layer's match head
        self.dim = lg.conf.descriptor_dim
        self.h = lg.conf.num_heads
        self.hd = self.dim // self.h

    def rotate_half(self, x):  # x: [b, h, k, hd]
        x = x.reshape(self.b, self.h, self.k, self.hd // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).reshape(self.b, self.h, self.k, self.hd)

    def rotary(self, freqs, t):
        return (t * freqs[0]) + (self.rotate_half(t) * freqs[1])

    def posenc(self, kpts):  # LearnableFourierPositionalEncoding, repeat_interleave written out
        proj = self.lg.posenc.Wr(kpts)  # [b, k, hd/2]
        emb = torch.stack([torch.cos(proj), torch.sin(proj)], 0).unsqueeze(-3)  # [2, b, 1, k, hd/2]
        half = emb.shape[-1]
        return emb.unsqueeze(-1).expand(2, self.b, 1, self.k, half, 2).reshape(2, self.b, 1, self.k, 2 * half)

    def heads(self, t):  # [b, k, dim] -> [b, h, k, hd]
        return t.reshape(self.b, self.k, self.h, self.hd).transpose(1, 2)

    def merge(self, t):  # [b, h, k, hd] -> [b, k, dim]
        return t.transpose(1, 2).reshape(self.b, self.k, self.dim)

    def self_block(self, blk, x, enc, key_valid):
        qkv = blk.Wqkv(x).reshape(self.b, self.k, self.h, self.hd, 3).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k = self.rotary(enc, q), self.rotary(enc, k)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :])
        msg = blk.out_proj(self.merge(ctx))
        return x + blk.ffn(torch.cat([x, msg], -1))

    def cross_block(self, blk, x0, x1, valid0, valid1):
        qk0, qk1 = self.heads(blk.to_qk(x0)), self.heads(blk.to_qk(x1))
        v0, v1 = self.heads(blk.to_v(x0)), self.heads(blk.to_v(x1))
        m0 = F.scaled_dot_product_attention(qk0, qk1, v1, attn_mask=valid1[:, None, None, :])
        m1 = F.scaled_dot_product_attention(qk1, qk0, v0, attn_mask=valid0[:, None, None, :])
        m0, m1 = blk.to_out(self.merge(m0)), blk.to_out(self.merge(m1))
        return x0 + blk.ffn(torch.cat([x0, m0], -1)), x1 + blk.ffn(torch.cat([x1, m1], -1))

    def forward(self, kpts0, kpts1, desc0, desc1, valid0, valid1):
        lg = self.lg
        v0, v1 = valid0 > 0.5, valid1 > 0.5
        d0, d1 = lg.input_proj(desc0), lg.input_proj(desc1)
        e0, e1 = self.posenc(kpts0), self.posenc(kpts1)
        for layer in lg.transformers[: self.n_layers]:
            d0 = self.self_block(layer.self_attn, d0, e0, v0)
            d1 = self.self_block(layer.self_attn, d1, e1, v1)
            d0, d1 = self.cross_block(layer.cross_attn, d0, d1, v0, v1)
        la = lg.log_assignment[self.n_layers - 1]
        m0, m1 = la.final_proj(d0), la.final_proj(d1)
        sim = torch.matmul(m0 / self.dim ** 0.25, (m1 / self.dim ** 0.25).transpose(1, 2))
        neg = torch.finfo(sim.dtype).min
        s0 = F.log_softmax(sim.masked_fill(~v1[:, None, :], neg), 2)
        s1 = F.log_softmax(sim.masked_fill(~v0[:, :, None], neg), 1)
        cert = F.logsigmoid(la.matchability(d0)) + F.logsigmoid(la.matchability(d1)).transpose(1, 2)
        return s0 + s1 + cert


def prepare(feats, k, add_scale_ori=True):
    """SIFT features dict (batch 1) -> fixed-size engine inputs and the real point count."""
    from lightglue.lightglue import normalize_keypoints
    kp = normalize_keypoints(feats["keypoints"].float(), feats["image_size"].float())
    if add_scale_ori:
        kp = torch.cat([kp, feats["scales"].float().unsqueeze(-1), feats["oris"].float().unsqueeze(-1)], -1)
    desc = feats["descriptors"].float()
    n = kp.shape[1]
    if n > k:
        raise ValueError(f"{n} points > engine size {k}")
    pad = k - n
    kp = F.pad(kp, (0, 0, 0, pad))
    desc = F.pad(desc, (0, 0, 0, pad))
    valid = torch.zeros(1, k, device=kp.device)
    valid[:, :n] = 1.0
    return kp.contiguous(), desc.contiguous(), valid, n


def matches_from_scores(scores, m, n, threshold):
    """Mutual best matches above threshold, as lightglue.lightglue.filter_matches, on the
    valid m x n block. Returns an (S, 2) tensor of (photo index, tile index)."""
    s = scores[:, :m, :n].float()
    max0, max1 = s.max(2), s.max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = torch.arange(m, device=s.device)[None] == m1.gather(1, m0)
    ms0 = torch.where(mutual0, max0.values.exp(), s.new_tensor(0))
    valid = mutual0 & (ms0 > threshold)
    i0 = torch.where(valid[0])[0]
    return torch.stack([i0, m0[0][valid[0]]], -1)


class LightGlueTRT:
    """Runs the TensorRT engine, with PyTorch CUDA tensors as buffers (same CUDA context as
    the rest of the pipeline)."""

    def __init__(self, engine_path, threshold=0.1, add_scale_ori=True):
        import tensorrt as trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.buf = {}
        for nm in names:
            shape = tuple(self.ctx.get_tensor_shape(nm))
            self.buf[nm] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(nm, self.buf[nm].data_ptr())
        self.out_name = [nm for nm in names if nm not in INPUTS][0]
        self.k = self.buf["kpts0"].shape[1]
        self.threshold, self.add_scale_ori = threshold, add_scale_ori
        self.stream = torch.cuda.current_stream()

    def set_photo(self, f0):
        kp, desc, valid, self.m = prepare(f0, self.k, self.add_scale_ori)
        self.buf["kpts0"].copy_(kp); self.buf["desc0"].copy_(desc); self.buf["valid0"].copy_(valid)

    def match(self, f1):
        kp, desc, valid, n = prepare(f1, self.k, self.add_scale_ori)
        self.buf["kpts1"].copy_(kp); self.buf["desc1"].copy_(desc); self.buf["valid1"].copy_(valid)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        return matches_from_scores(self.buf[self.out_name], self.m, n, self.threshold)


def load_lightglue():
    from lightglue import LightGlue
    return LightGlue(features="sift").eval()


def export(args):
    sys.path.insert(0, os.getcwd())
    lg = load_lightglue()
    k, b = args.points, args.batch
    model = LightGlueStatic(lg, k, b, args.layers).eval()
    torch.manual_seed(0)
    dummy = (torch.rand(b, k, 4) * 2 - 1, torch.rand(b, k, 4) * 2 - 1, torch.randn(b, k, 128), torch.randn(b, k, 128),
             (torch.arange(k) < k - 100).float()[None].repeat(b, 1), (torch.arange(k) < k - 200).float()[None].repeat(b, 1))
    with torch.no_grad():
        torch.onnx.export(model, dummy, os.path.expanduser(args.out), input_names=list(INPUTS), output_names=["scores"],
                          opset_version=17, do_constant_folding=True)
    print(f"wrote {args.out} ({os.path.getsize(os.path.expanduser(args.out)) / 1e6:.1f} MB)")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.expanduser(args.out), providers=["CPUExecutionProvider"])
        out = sess.run(["scores"], {nm: t.numpy() for nm, t in zip(INPUTS, dummy)})[0]
        with torch.no_grad():
            ref = model(*dummy).numpy()
        m, n = k - 100, k - 200
        print(f"ONNX vs PyTorch on the valid block: max abs diff {np.abs(out[:, :m, :n] - ref[:, :m, :n]).max():.2e}")
    except ImportError:
        print("onnxruntime not installed; skipped the ONNX check")


def check(args):
    """Real photo/tile pairs: static model (padded, masked) vs LightGlue with early stopping
    and pruning off. Matches should be identical up to float rounding."""
    import glob
    import json
    from PIL import Image
    sys.path.insert(0, os.getcwd())
    from data_loading.earthloc_loader import parse_geotile_filename
    from lightglue import SIFT
    from lightglue.utils import numpy_image_to_torch
    data = os.path.expanduser(args.data)
    q = json.load(open(os.path.join(data, "queries.json")))
    meta = json.load(open(os.path.join(data, "db_tiles.json")))
    by_id = {t.tile_id: t for t in map(parse_geotile_filename,
                                       glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))}
    sift = SIFT(max_num_keypoints=1024).eval()
    lg = load_lightglue()
    lg.conf.depth_confidence = lg.conf.width_confidence = -1
    static = LightGlueStatic(lg, args.points).eval()

    def feats(path):
        with torch.no_grad():
            return sift.extract(numpy_image_to_torch(np.array(Image.open(path).convert("RGB"))), resize=512)

    same, total = 0, 0
    for qm in q[: args.photos]:
        f0 = feats(os.path.join(data, "queries", qm["file"]))
        for idx in qm["positives"][:2] + [0]:
            f1 = feats(by_id[meta[idx]["tile_id"]].image_path)
            with torch.no_grad():
                ref = lg({"image0": f0, "image1": f1})["matches"][0]
                kp0, d0, v0, m = prepare(f0, args.points)
                kp1, d1, v1, n = prepare(f1, args.points)
                got = matches_from_scores(static(kp0, kp1, d0, d1, v0, v1), m, n, lg.conf.filter_threshold)
            a, b = {tuple(x) for x in ref.tolist()}, {tuple(x) for x in got.tolist()}
            same += int(a == b)
            total += 1
            print(f"  {qm['file'][-30:]} tile {idx}: reference {len(a)} matches, static {len(b)}, overlap {len(a & b) / max(len(a | b), 1):.3f}")
    print(f"identical match sets: {same}/{total}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--points", type=int, default=1024)
    e.add_argument("--batch", type=int, default=1, help="tile pairs per engine call (photo inputs repeated)")
    e.add_argument("--layers", type=int, default=None, help="use only the first N of LightGlue's 9 layers")
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--tiles-dir", required=True)
    c.add_argument("--photos", type=int, default=3)
    c.add_argument("--points", type=int, default=1024)
    args = ap.parse_args()
    export(args) if args.cmd == "export" else check(args)


if __name__ == "__main__":
    main()
