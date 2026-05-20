import torch
from model import ChessCNN

MODEL_PATH = "model.pt"
ONNX_PATH = "chess_model.onnx"

model = ChessCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 13, 8, 8)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["value", "policy"],   # ✔ two outputs
        dynamic_axes={
            "input": {0: "batch_size"},
            "value": {0: "batch_size"},
            "policy": {0: "batch_size"}
        },
        dynamo=False
    )

print("✅ ONNX Export Complete")