import torch
from train_code.model.deeplab import get_deeplab_model

def test_get_deeplab_model():
    model = get_deeplab_model(num_classes=3, device=torch.device("cpu"))
    model.eval()
    assert hasattr(model, "classifier")
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    assert "out" in out
    assert out["out"].shape[1] == 3