import os
import types
import torch
import pytest

from pruner import prune_main

class DummyArgs:
    model = "dummy_model.pth"
    datapath = None
    target_prune_rate = 0.2
    iterative_steps = 1
    epochs = 1
    max_map_drop = 0.5

class MockConfig:
    def __init__(self, dataset_path_override=None):
        self.dataset_path = dataset_path_override or "dummy_data_path"
        self.num_classes = 3
        self.input_size = (16, 16) # Keep it small for tests
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For val_loader in prune_main
        self.val_dataset_path = "dummy_val_images"
        self.val_names = ["img1", "img2"]
        self.val_mask_path = "dummy_val_masks"
        self.batch_size = 1 # Smaller batch size for testing
        self.num_workers = 0

        # For criterion in prune_main
        self.loss_type = "cross_entropy"
        self.ce_weight = 1.0
        self.dice_weight = 1.0

        # For finetune (even though mocked, its signature might be checked)
        self.train_dataset_path = "dummy_train_images"
        self.train_names = ["train_img1", "train_img2"]
        self.train_mask_path = "dummy_train_masks"
        self.learning_rate = 1e-4

        # For saving paths, not strictly needed if os.makedirs is fully mocked
        # but good for completeness if any path logic is tested.
        self.date_str = "20230101"
        self._run_id = 1
        self._base_model_name = f"test_model_{self.date_str}_{self._run_id}"

    @property
    def run_id(self): return self._run_id
    @property
    def base_model_name(self): return self._base_model_name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for device test")
def test_prune_main_runs_successfully(monkeypatch, tmp_path):
    args = DummyArgs()
    args.datapath = str(tmp_path / "dataset")
    os.makedirs(args.datapath, exist_ok=True)

    # Create a dummy model file as expected by prune_main
    dummy_model_path = tmp_path / args.model
    torch.save({"conv.weight": torch.randn(3,3,3,3), "conv.bias": torch.randn(3)}, dummy_model_path)
    args.model = str(dummy_model_path) # Ensure full path is used

    # --- Mock external dependencies ---
    monkeypatch.setattr("pruner.Config", MockConfig)
    monkeypatch.setattr("pruner.get_lraspp_model", lambda *a, **kw: torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1)))
    monkeypatch.setattr("torch.load", lambda *a, **kw: {"0.weight": torch.randn(3, 3, 3, 3), "0.bias": torch.randn(3)})
    monkeypatch.setattr("torch.save", lambda *a, **kw: None)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok=False: None)
    monkeypatch.setattr("os.path.getsize", lambda path: 1024 * 1024) # Dummy file size 1MB

    dummy_dataset = [(torch.randn(3, 16, 16), torch.randint(0, 3, (16, 16))) for _ in range(2)]
    monkeypatch.setattr("pruner.LaRSDataset", lambda *a, **kw: dummy_dataset)
    monkeypatch.setattr("pruner.DataLoader", lambda dataset, **kw: dataset)
    monkeypatch.setattr("pruner.get_loss_function", lambda *a, **kw: torch.nn.CrossEntropyLoss())
    monkeypatch.setattr("pruner.validate", lambda *a, **kw: (0.9, 0.8, 0.1))
    monkeypatch.setattr("pruner.finetune", lambda model, config, epochs, **kw: ([0.1]*epochs, [0.5]*epochs, [0.6]*epochs, [0.7]*epochs, [0.8]*epochs))
    monkeypatch.setattr("pruner.tp", types.SimpleNamespace(utils=types.SimpleNamespace(count_ops_and_params=lambda m, x: (1e6, 1e4))))

    monkeypatch.setattr("pruner.prune_utils.l1_unstructured", lambda module, name, amount: None)
    monkeypatch.setattr("pruner.prune_utils.is_pruned", lambda module: True) # Assume already pruned for remove to proceed
    monkeypatch.setattr("pruner.prune_utils.remove", lambda module, name: None)

    monkeypatch.setattr("pruner.gc.collect", lambda: None)
    monkeypatch.setattr("pruner.torch.cuda.empty_cache", lambda: None)

    monkeypatch.setattr("pruner.save_metrics_plot", lambda *a, **kw: None)
    monkeypatch.setattr("pruner.save_metrics_to_csv", lambda *a, **kw: None)

    # Call the target function
    prune_main(args)