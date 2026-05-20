import onnx
import onnxruntime as ort
import numpy as np

ONNX_PATH = "chess_model.onnx"

# 1️⃣ Check ONNX structure
model = onnx.load(ONNX_PATH)
onnx.checker.check_model(model)
print("✅ ONNX model structure is valid.")

# 2️⃣ Run inference test
session = ort.InferenceSession(ONNX_PATH)

dummy_input = np.random.randn(1, 13, 8, 8).astype(np.float32)

outputs = session.run(None, {"input": dummy_input})

print("✅ ONNX inference successful.")
print("Output:", outputs[0])
