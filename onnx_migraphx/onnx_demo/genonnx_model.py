import torch
import torch.nn as nn

# A simple model that adds a scalar to a tensor
class AddScalarModel(nn.Module):
    def forward(self, tensor_input, scalar_input):
        # The ONNX exporter will treat the second input as a tensor,
        # but the C++ code correctly handles it as a scalar.
        return tensor_input + scalar_input

# Instantiate the model
model = AddScalarModel()
model.eval()

# Dummy inputs that match the C++ code's expectations
# Batch size is dynamic, so we can use any valid size here, e.g., 1
tensor = torch.ones(1, 3, 4, 5, dtype=torch.uint8)
scalar = torch.ones(1, dtype=torch.uint8) # Represents the scalar to be added

# Export to ONNX
torch.onnx.export(
    model,
    (tensor, scalar),
    "add_scalar_test1.onnx", # Save in the correct location for your C++ app
    input_names=["0", "1"], # Match the names used in the C++ code
    output_names=["output"],
    opset_version=12,
    dynamic_axes={
        "0": {0: "batch_size"}, # Make the batch dimension of input "0" dynamic
    }
)

print("Successfully created 'add_scalar_test.onnx'")