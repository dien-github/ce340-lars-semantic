import torch
from model.deeplab import get_lraspp_model
from config import Config

config = Config()
model_path = "/home/grace/Documents/ce340-lars-semantic/best_model_20250608_1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_lraspp_model(num_classes=config.num_classes, device=device)
for p in model.parameters():
    p.requires_grad = True
print(f"Loading model from: {model_path} ...")
state_dict = torch.load(model_path, map_location=device)
new_state_dict = {
    k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
    for k, v in state_dict.items()
}
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
if missing or unexpected:
    print(f"[Warning] Missing keys when loading model: {missing}")
    print(f"[Warning] Unexpected keys: {unexpected}")
print("Model loaded successfully.")

model.eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 320, 320)

# Export the model to ONNX format
torch.onnx.export(
    model.cpu(),
    dummy_input,
    "20250608_1.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)
print("Model exported to 20250608_1.onnx")
