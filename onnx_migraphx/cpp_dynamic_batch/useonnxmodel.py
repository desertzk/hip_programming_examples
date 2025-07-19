# save this as run_add_ort.py and run: python run_add_ort.py
import onnxruntime as ort
import numpy as np

# 1) Load the model
sess = ort.InferenceSession("add.onnx")

# 2) Prepare two random inputs of shape [2,3,4,5]
a = np.random.rand(2,3,4,5).astype(np.float32)
b = np.random.rand(2,3,4,5).astype(np.float32)

# 3) Run
outs = sess.run(
    output_names=None,         # returns all outputs
    input_feed={'0': a, '1': b}
)

# 4) Verify
c = outs[0]
print("shape:", c.shape)  # (2,3,4,5)
print(c)
# should equal a + b (up to FP rounding)
print("max abs error:", np.max(np.abs(c - (a + b))))
