import onnx
from onnx import helper, TensorProto, OperatorSetIdProto
import numpy as np
import onnxruntime as ort

# 1) Define inputs with valid names
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])

# 2) Define output
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [2, 3])

# 3) Single Add node that takes X,Y → Z
node_def = helper.make_node(
    'Add',      # op type
    ['X', 'Y'], # inputs
    ['Z'],      # outputs
)

# 4) Build graph
graph_def = helper.make_graph(
    [node_def],
    'add_graph',
    [X, Y],
    [Z]
)

# 5) Specify an explicit opset import (<=22)
opset = OperatorSetIdProto()
opset.domain = ""      # default (ai.onnx)
opset.version = 22     # pick any supported opset ≤ 22

# 6) Build model, then clamp IR version to 10
model_def = helper.make_model(
    graph_def,
    producer_name='example',
    opset_imports=[opset]
)
model_def.ir_version = 10

# 7) Save
onnx.save(model_def, 'add.onnx')
print("Saved add.onnx (IR=10, opset=22)")

# 8) Load with ONNX Runtime
sess = ort.InferenceSession("add.onnx")

# 9) Prepare two random inputs
a = np.random.rand(2,3).astype(np.float32)
b = np.random.rand(2,3).astype(np.float32)

# 10) Run
outs = sess.run(
    output_names=None,         # returns all outputs
    input_feed={'X': a, 'Y': b}
)

# 11) Verify
c = outs[0]
print("shape:", c.shape)  
print("max abs error:", np.max(np.abs(c - (a + b))))
