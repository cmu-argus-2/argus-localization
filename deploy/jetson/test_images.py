"""Run real JPEGs through an engine: decode, resize, ImageNet-normalize, embed.
Run on the Jetson with sample queries in samples/.

    python3 test_images.py nano_fp32.trt

Cross-similarity well below 1.0 means the model discriminates between images.
Near 1.0 everywhere would mean preprocessing is broken.
"""
import glob, sys
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "nano_fp32.trt"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
in_shape  = tuple(ctx.get_tensor_shape(in_name))
out_shape = tuple(ctx.get_tensor_shape(out_name))

# allocate once and reuse; allocating per image leaks and segfaults at teardown
host_in  = np.empty(in_shape,  dtype=np.float32)
host_out = np.empty(out_shape, dtype=np.float32)
d_in  = cuda.mem_alloc(host_in.nbytes)
d_out = cuda.mem_alloc(host_out.nbytes)
stream = cuda.Stream()
ctx.set_tensor_address(in_name, int(d_in))
ctx.set_tensor_address(out_name, int(d_out))

def embed(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2,0,1) / 255.0
    host_in[0] = (x - MEAN) / STD
    cuda.memcpy_htod_async(d_in, host_in, stream)
    ctx.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(host_out, d_out, stream)
    stream.synchronize()
    return host_out[0].copy()

files = sorted(glob.glob("samples/*.jpg"))
print(f"engine: {ENGINE}   images: {len(files)}\n")
D = np.stack([embed(p) for p in files])

S = D @ D.T
off = S[~np.eye(len(D), dtype=bool)]
print(f"all L2 norms ~1.0 : {np.allclose(np.linalg.norm(D,axis=1), 1.0, atol=1e-3)}")
print(f"no NaNs           : {not np.isnan(D).any()}")
print(f"cross-similarity  : min {off.min():.3f}  mean {off.mean():.3f}  max {off.max():.3f}")

del ctx, engine
d_in.free(); d_out.free()
