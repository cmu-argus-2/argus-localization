"""Compare two engines on the SAME input. Used to reject FP16. Run on the Jetson.

    python3 compare_precision.py        # compares nano_fp32.trt vs nano_fp16.trt
"""
import numpy as np, tensorrt as trt, pycuda.autoinit, pycuda.driver as cuda

def run(engine_path, x):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    i, o = engine.get_tensor_name(0), engine.get_tensor_name(1)
    out = np.empty(tuple(ctx.get_tensor_shape(o)), dtype=np.float32)
    d_in, d_out = cuda.mem_alloc(x.nbytes), cuda.mem_alloc(out.nbytes)
    s = cuda.Stream()
    cuda.memcpy_htod_async(d_in, x, s)
    ctx.set_tensor_address(i, int(d_in)); ctx.set_tensor_address(o, int(d_out))
    ctx.execute_async_v3(s.handle)
    cuda.memcpy_dtoh_async(out, d_out, s); s.synchronize()
    return out

rng = np.random.default_rng(1234)
x = np.ascontiguousarray(rng.standard_normal((1,3,224,224)).astype(np.float32))

a = run("nano_fp32.trt", x)
b = run("nano_fp16.trt", x)
cos = float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b)))
print("SAME input, both engines")
print(f"  max abs diff : {np.abs(a-b).max():.6f}")
print(f"  cosine sim   : {cos:.6f}   (want > 0.999)")
print("\n" + ("FP16 is safe to use." if cos > 0.999 else "FP16 diverges - stay on FP32."))
