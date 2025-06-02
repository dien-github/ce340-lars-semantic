import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLoss(nn.Module):
    def __init__(self, num_classes=1, reduction='mean'):
        """
        num_classes: số class (đối với segmentation đa lớp, ta sẽ xem đầu ra softmax one-hot)
        reduction: 'mean' hoặc 'sum'
        """
        super(LaplacianLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

        # Tạo kernel Laplacian 3x3 (8-connected)
        # [ 0 -1  0]
        # [-1  4 -1]
        # [ 0 -1  0]
        lap = torch.tensor([[0., -1., 0.],
                            [-1., 4., -1.],
                            [0., -1., 0.]], dtype=torch.float32)
        # Vì ta sẽ apply conv2d lên nhiều channel (num_classes), nên đẩy vào dạng (out_ch, in_ch, k, k)
        # Ở đây out_ch = in_ch = num_classes, kernel tách kênh (depthwise), nên đặt groups=num_classes khi conv.
        self.register_buffer('kernel', lap.unsqueeze(0).unsqueeze(0).repeat(self.num_classes, 1, 1, 1))
        # Ví dụ: nếu num_classes=3, kernel.shape = (3,1,3,3)

    def forward(self, logits, target):
        """
        logits: tensor shape (B, C, H, W), chưa qua softmax (đối với đa lớp), 
                hoặc (B,1,H,W) nếu nhị phân.
        target: tensor shape (B, H, W) chứa nhãn integer [0..C-1] (đối với đa lớp),
                hoặc (B, H, W) nhị phân {0,1} (nếu num_classes=1).
        """
        # device=logits.device
        # kernel = self.kernel.to(device)


        # 1. Lấy xác suất dự đoán (softmax hoặc sigmoid) tuỳ num_classes
        if self.num_classes > 1:
            prob = F.softmax(logits, dim=1)  # (B, C, H, W)
            # Chuyển target về one-hot: (B, C, H, W)
            target_onehot = F.one_hot(target.long(), num_classes=self.num_classes)  # (B, H, W, C)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        else:
            prob = torch.sigmoid(logits)  # (B,1,H,W) giả sử logits có channel=1
            target_onehot = target.unsqueeze(1).float()  # (B,1,H,W)

        # 2. Tính Laplacian của prob và của target_onehot
        #    Sử dụng conv2d với groups=num_classes để apply depthwise convolution
        laplacian_pred = F.conv2d(prob, self.kernel, padding=1, groups=self.num_classes)      # (B,C,H,W)
        laplacian_gt   = F.conv2d(target_onehot, self.kernel, padding=1, groups=self.num_classes)  # (B,C,H,W)

        # 3. Tính loss (ở đây dùng L1: |L(pred)-L(gt)|)
        diff = torch.abs(laplacian_pred - laplacian_gt)

        if self.reduction == 'mean':
            return diff.mean()
        elif self.reduction == 'sum':
            return diff.sum()
        else:
            # none
            return diff


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs = torch.softmax(inputs, dim=1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, lap_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.lap = LaplacianLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lap_weight = lap_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets.long() if self.num_classes > 1 else targets.unsqueeze(1).float())
        dice_loss = self.dice(inputs, targets)
        lap_loss = self.lap(inputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss + self.lap_weight * lap_loss


def get_loss_function(loss_type, **kwargs):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "combined":
        ce_weight = kwargs.get("ce_weight", 1.0)
        dice_weight = kwargs.get("dice_weight", 1.0)
        lap_weight = kwargs.get("lap_weight", 1.0)
        return CombinedLoss(num_classes=kwargs.get("num_classes", 1),ce_weight=ce_weight, dice_weight=dice_weight, lap_weight=lap_weight)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Use 'cross_entropy' or 'combined'."
        )
