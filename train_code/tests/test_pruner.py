import types
import torch
import pytest

from pruner import prune

class DummyArgs:
    model = "dummy_model.pth"
    target_prune_rate = 0.2
    iterative_steps = 1
    epochs = 1
    max_map_drop = 0.5

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for device test")
def test_prune_runs(monkeypatch, tmp_path):
    # Patch model loading and saving to avoid file I/O
    monkeypatch.setattr("pruner.get_lraspp_model", lambda *a, **kw: torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1)))
    monkeypatch.setattr("torch.load", lambda *a, **kw: {"0.weight": torch.randn(3, 3, 3, 3)})
    monkeypatch.setattr("torch.save", lambda *a, **kw: None)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("pruner.LaRSDataset", lambda *a, **kw: [ (torch.randn(3, 16, 16), torch.randint(0, 3, (16, 16))) for _ in range(2) ])
    monkeypatch.setattr("pruner.DataLoader", lambda dataset, **kw: dataset)
    monkeypatch.setattr("pruner.get_loss_function", lambda *a, **kw: torch.nn.CrossEntropyLoss())
    monkeypatch.setattr("pruner.validate", lambda *a, **kw: (0.9, 0.8, 0.1))
    monkeypatch.setattr("pruner.finetune", lambda *a, **kw: None)
    monkeypatch.setattr("pruner.tp", types.SimpleNamespace(utils=types.SimpleNamespace(count_ops_and_params=lambda m, x: (1e6, 1e4))))
    args = DummyArgs()
    prune(args)