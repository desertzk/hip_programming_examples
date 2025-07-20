import onnx

# Load the problematic model
onnx_model = onnx.load("add_scalar_test.onnx")

# The ONNX library performs checks and can help reorder.
# Simply saving it again can resolve the issue.
onnx.checker.check_model(onnx_model) # This would likely fail on your original model

# Save the model to a new file. The library will serialize it in a valid order.
onnx.save(onnx_model, "your_fixed_model.onnx")

print("Model has been re-saved. Try loading 'your_fixed_model.onnx'")