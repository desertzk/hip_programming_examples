import numpy as np
from numba import hip

@hip.jit
def vector_add(a, b, c):
    tid = hip.grid(1)  # Global thread index [[6]]
    size = c.size
    
    if tid < size:
        c[tid] = a[tid] + b[tid]

# Configuration parameters
n = 1000  
threads_per_block = 256  # Common block size for GPU kernels [[6]]
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block  # Grid calculation [[2]]

# Initialize host arrays
a = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
b = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
c = np.empty_like(a)

# Explicit data transfer to GPU (required for HIP)
d_a = hip.to_device(a)
d_b = hip.to_device(b)
d_c = hip.device_array_like(c)

# Kernel launch with grid/block configuration [[6]]
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy result back to host
c_gpu = d_c.copy_to_host()

# Validation
print("First 5 elements:", c_gpu[:5])
print("Max error:", np.max(np.abs(c_gpu - (a + b))))
