import torch
import pytest
from utils.losses import DiceLoss, CombinedLoss, LaplacianLoss, get_loss_function


def test_dice_loss_shape_and_value():
    loss_fn = DiceLoss()
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert 0 <= loss.item() <= 1


def test_combined_loss_shape_and_value():
    loss_fn = CombinedLoss(
        num_classes=3, ce_weight=1.0, dice_weight=1.0, lap_weight=1.0
    )
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_laplacian_loss_shape_and_value():
    loss_fn = LaplacianLoss(num_classes=3)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_get_loss_function():
    ce = get_loss_function("cross_entropy")
    assert hasattr(ce, "__call__")
    combined = get_loss_function(
        "combined", num_classes=3, ce_weight=1.0, dice_weight=1.0, lap_weight=1.0
    )
    assert hasattr(combined, "__call__")
    with pytest.raises(ValueError):
        get_loss_function("unknown")
