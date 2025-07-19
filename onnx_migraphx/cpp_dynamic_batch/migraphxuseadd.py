# save this as run_with_migraphx.py and run it: python run_with_migraphx.py
import migraphx
import numpy as np

# --- MIGraphX Inference ---

# 1) Parse the ONNX model file.
#    We are using 'add_robust.onnx' which was created to be compatible.
try:
    model = migraphx.parse_onnx("add_robust.onnx")
    print("Successfully parsed add_robust.onnx")
except Exception as e:
    print(f"Error parsing ONNX file: {e}")
    print("Please ensure 'add_robust.onnx' is in the same directory.")
    exit()

# 2) Compile the model for the target hardware.
#    Use migraphx.get_target("gpu") for AMD GPUs.
#    Use migraphx.get_target("cpu") to run on the CPU.
print("Compiling model for GPU...")
model.compile(migraphx.get_target("gpu"))
print("Model compiled successfully.")

# 3) Prepare the input data.
#    The shape and type must match the model's expectations.
#    The dictionary keys ('INPUT_A', 'INPUT_B') must match the input names in the ONNX graph.
a = np.random.rand(2, 3, 4, 5).astype(np.float32)
b = np.random.rand(2, 3, 4, 5).astype(np.float32)

# 4) Run inference.
#    The inputs are passed as a dictionary where keys are input names
#    and values are the numpy arrays.
print("Running inference...")
results = model.run({
    'INPUT_A': a,
    'INPUT_B': b
})
print("Inference complete.")

# 5) Process and verify the output.
#    The result is a migraphx.argument object. To use it with numpy,
#    it needs to be converted back to a numpy array.
#    Since the model has one output, we access the first element of the results.
c = np.array(results[0])

print("\n--- Verifying Output ---")
print("Output shape:", c.shape)

# The result should equal a + b (up to floating-point rounding)
max_error = np.max(np.abs(c - (a + b)))
print("Max absolute error:", max_error)

if max_error < 1e-6:
    print("Verification successful: The output matches the expected sum.")
else:
    print("Verification failed: The output does not match the expected sum.")

