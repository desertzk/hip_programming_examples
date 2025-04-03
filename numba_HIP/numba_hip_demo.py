import numpy as np
from numba import jit
import numba.hip as hip
import time

# ----------------------------------------
# Step 1: Check if HIP is available
# ----------------------------------------
if not hip.is_available():
    raise RuntimeError("HIP/ROCm is not available on this system.")

print("HIP devices found:", hip.detect())

# ----------------------------------------
# Step 2: Generate sample data (CPU)
# ----------------------------------------
N = 10_000_000  # 10 million elements
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
out = np.zeros_like(a)

# ----------------------------------------
# Step 3: Copy data to GPU (HIP device)
# ----------------------------------------
a_d = hip.to_device(a)
b_d = hip.to_device(b)
out_d = hip.device_array_like(out)

# ----------------------------------------
# Step 4: Define HIP kernel
# ----------------------------------------
@jit(target='hip')  # Compile for AMD GPU
def add_kernel(a, b, out):
    i = hip.get_global_id(0)  # Thread index in 1D grid
    if i < a.size:
        out[i] = a[i] + b[i]

# ----------------------------------------
# Step 5: Launch kernel on GPU
# ----------------------------------------
# Configure grid/block dimensions
threads_per_block = 256
blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block

# Time the GPU execution
start = time.time()
add_kernel[blocks_per_grid, threads_per_block](a_d, b_d, out_d)
hip.synchronize()  # Wait for kernel to finish
gpu_time = time.time() - start

# ----------------------------------------
# Step 6: Copy result back to CPU
# ----------------------------------------
out_gpu = out_d.copy_to_host()

# ----------------------------------------
# Step 7: Verify correctness (CPU check)
# ----------------------------------------
# Compute on CPU for comparison
start = time.time()
out_cpu = a + b
cpu_time = time.time() - start

# Check if GPU and CPU results match
tolerance = 1e-6
assert np.allclose(out_gpu, out_cpu, atol=tolerance), "GPU result mismatch!"

# ----------------------------------------
# Step 8: Print performance stats
# ----------------------------------------
print(f"GPU time: {gpu_time:.5f} seconds")
print(f"CPU time: {cpu_time:.5f} seconds")
print("Speedup:  {:.1f}x".format(cpu_time / gpu_time))
print("Result verified!")
