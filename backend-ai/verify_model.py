import onnx

model = onnx.load("chess_model.onnx")

onnx.checker.check_model(model)

print("✅ ONNX model is valid")

print("Input shape:")
print(model.graph.input[0])

print("\nOutputs:")
for out in model.graph.output:
    print(out.name)