# build_add_model.py

import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort

# 1) Define inputs with valid names
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4, 5])

# 2) Define output
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [2, 3, 4, 5])

# 3) Single Add node that takes X,Y → Z
node_def = helper.make_node(
    'Add',      # op type
    ['X', 'Y'], # inputs
    ['Z'],      # outputs
)

# 4) Build graph & model
graph_def = helper.make_graph(
    [node_def],
    'add_graph',
    [X, Y],
    [Z]
)
model_def = helper.make_model(graph_def, producer_name='example')
onnx.save(model_def, 'add.onnx')
print("Saved add.onnx")

# 5) Load with ONNX Runtime
sess = ort.InferenceSession("add.onnx")

# 6) Prepare two random inputs
a = np.random.rand(2,3,4,5).astype(np.float32)
b = np.random.rand(2,3,4,5).astype(np.float32)

# 7) Run
outs = sess.run(
    output_names=None,         # returns all outputs
    input_feed={'X': a, 'Y': b}
)

# 8) Verify
c = outs[0]
print("shape:", c.shape)  # (2,3,4,5)
print("max abs error:", np.max(np.abs(c - (a + b))))
