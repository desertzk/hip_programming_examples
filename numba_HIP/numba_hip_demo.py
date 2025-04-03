import numpy as np
from numba import jit
import numba.hip as hip
import time

# ----------------------------------------
# Step 1: Check HIP availability
# ----------------------------------------
if not hip.is_available():
    raise RuntimeError("HIP/ROCm is not available on this system.")

# ----------------------------------------
# Step 2: Generate sample data
# ----------------------------------------
N = 10_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
out = np.zeros_like(a)

# ----------------------------------------
# Step 3: Copy data to GPU
# ----------------------------------------
a_d = hip.to_device(a)
b_d = hip.to_device(b)
out_d = hip.device_array_like(out)

# ----------------------------------------
# Step 4: HIP kernel (corrected)
# ----------------------------------------
@hip.jit
def add_kernel(a, b, out):
    i = hip.blockIdx.x * hip.blockDim.x + hip.threadIdx.x
    if i < a.size:
        out[i] = a[i] + b[i]

# ----------------------------------------
# Step 5: Launch kernel (with timing fix)
# ----------------------------------------
threads_per_block = 256
blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block

# Create a stream for execution
stream = hip.stream()
start = time.time()

# Launch kernel with explicit stream
add_kernel[blocks_per_grid, threads_per_block, stream](a_d, b_d, out_d)

# Synchronize using the stream instead of global context
stream.synchronize()

gpu_time = time.time() - start

# ----------------------------------------
# Step 6: Copy results back
# ----------------------------------------
out_gpu = out_d.copy_to_host()

# ----------------------------------------
# Step 7: Verify correctness
# ----------------------------------------
out_cpu = a + b
tolerance = 1e-6
print(out_gpu)

assert np.allclose(out_gpu, out_cpu, atol=tolerance), "Mismatch!"

# ----------------------------------------
# Step 8: Print stats
# ----------------------------------------
print(f"GPU time: {gpu_time:.5f}s")
print("Result verified!")
