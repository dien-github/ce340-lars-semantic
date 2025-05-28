import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        targets_one_hot = F.onehot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs = torch.softmax(inputs, dim=1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loss_function(loss_type, **kwargs):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "combined":
        ce_weight = kwargs.get("ce_weight", 1.0)
        dice_weight = kwargs.get("dice_weight", 1.0)
        return CombinedLoss(ce_weight=ce_weight, dice_weight=dice_weight)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Use 'cross_entropy' or 'combined'."
        )
