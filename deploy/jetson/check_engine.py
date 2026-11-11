"""Sanity-check a TensorRT engine: shape, L2 norm, NaNs. Run on the Jetson.

    python3 check_engine.py nano_fp32.trt

Saves <engine>_output.npy so the result can be cross-checked against PyTorch
on the workstation (see RUNBOOK "Verification chain").
"""
import sys
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

engine_path = sys.argv[1]

logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

in_name = engine.get_tensor_name(0)
out_name = engine.get_tensor_name(1)
in_shape = tuple(ctx.get_tensor_shape(in_name))
out_shape = tuple(ctx.get_tensor_shape(out_name))
print(f"engine : {engine_path}")
print(f"input  : {in_name} {in_shape}")
print(f"output : {out_name} {out_shape}")

rng = np.random.default_rng(1234)
x = np.ascontiguousarray(rng.standard_normal(in_shape).astype(np.float32))
out = np.empty(out_shape, dtype=np.float32)

d_in = cuda.mem_alloc(x.nbytes)
d_out = cuda.mem_alloc(out.nbytes)
stream = cuda.Stream()
cuda.memcpy_htod_async(d_in, x, stream)
ctx.set_tensor_address(in_name, int(d_in))
ctx.set_tensor_address(out_name, int(d_out))
ctx.execute_async_v3(stream.handle)
cuda.memcpy_dtoh_async(out, d_out, stream)
stream.synchronize()

norms = np.linalg.norm(out, axis=1)
print(f"  L2 norm      : {norms}   (want 1.0)")
print(f"  NaN present  : {bool(np.isnan(out).any())}")
print(f"  all zeros    : {bool((out == 0).all())}")
print(f"  min / max    : {out.min():.6f} / {out.max():.6f}")

ok = (not np.isnan(out).any()) and (not (out == 0).all()) and np.allclose(norms, 1.0, atol=1e-3)
print("\n" + ("PASS: engine output is well-formed." if ok else "FAIL: output is not sane."))
np.save(engine_path.replace(".trt", "_output.npy"), out)
raise SystemExit(0 if ok else 1)
