import os
import argparse
from pathlib import Path
import tempfile

import pytest
import torch
import numpy as np
from PIL import Image

# Import the `main` function from your script
from main import main


@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Tạo cấu trúc thư mục và file hình ảnh/mask giả lập để main.py có thể chạy.
    """
    root = tmp_path / "LaRS_dataset"

    # Định nghĩa các folder theo Config:
    # train images:    LaRS_dataset/lars_v1.0.0_images/train/images
    # train masks:     LaRS_dataset/lars_v1.0.0_annotations/train/semantic_masks
    # val images:      LaRS_dataset/lars_v1.0.0_images/val/images
    # val masks:       LaRS_dataset/lars_v1.0.0_annotations/val/semantic_masks
    imgs_train = root / "lars_v1.0.0_images" / "train" / "images"
    masks_train = root / "lars_v1.0.0_annotations" / "train" / "semantic_masks"
    imgs_val = root / "lars_v1.0.0_images" / "val" / "images"
    masks_val = root / "lars_v1.0.0_annotations" / "val" / "semantic_masks"

    # Tạo tất cả các folder
    for d in (imgs_train, masks_train, imgs_val, masks_val):
        d.mkdir(parents=True, exist_ok=True)

    # Tạo một ảnh RGB 320x320 màu đen để dùng làm image và mask
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    dummy_mask = np.zeros((320, 320), dtype=np.uint8)  # mask chỉ 1 channel

    # Lưu dưới dạng 1.jpg, 1.png trong train/val
    img_train_path = imgs_train / "1.jpg"
    mask_train_path = masks_train / "1.png"
    Image.fromarray(dummy_img).save(img_train_path)
    Image.fromarray(dummy_mask).save(mask_train_path)

    img_val_path = imgs_val / "1.jpg"
    mask_val_path = masks_val / "1.png"
    Image.fromarray(dummy_img).save(img_val_path)
    Image.fromarray(dummy_mask).save(mask_val_path)

    # Tạo file image_list.txt cho train và val (chỉ chứa "1")
    list_train = root / "lars_v1.0.0_images" / "train" / "image_list.txt"
    list_val = root / "lars_v1.0.0_images" / "val" / "image_list.txt"
    with open(list_train, "w") as f:
        f.write("1\n")
    with open(list_val, "w") as f:
        f.write("1\n")

    return str(root)


def test_main_runs_without_error(dummy_dataset, monkeypatch):
    """
    Kiểm tra xem main(args) có chạy xong mà không lỗi trên dummy dataset.
    Chúng ta sẽ monkeypatch các hàm nặng để không thực sự train.
    """

    # 1. Monkeypatch các thành phần “nặng” để trả về giá trị giả lập
    # — Model: trả về một module rất nhẹ chỉ gồm Conv2d
    monkeypatch.setattr(
        "main.get_deeplab_model",
        lambda num_classes, device: torch.nn.Conv2d(3, num_classes, kernel_size=1),
    )
    monkeypatch.setattr(
        "main.get_lraspp_model",
        lambda num_classes,
        device,
        freeze_layers=None,
        unfreeze_layers=None: torch.nn.Conv2d(3, num_classes, kernel_size=1),
    )

    # — train_one_epoch trả về (time_taken, train_loss)
    monkeypatch.setattr(
        "main.train_one_epoch",
        lambda model,
        loader,
        criterion,
        optimizer,
        device,
        scaler,
        epoch,
        scheduler=None: (0.1, 0.0),
    )
    # — validate trả về (val_accuracy, val_miou, val_loss)
    monkeypatch.setattr(
        "main.validate",
        lambda model, loader, criterion, device, num_classes, epoch: (1.0, 1.0, 0.0),
    )

    # — Các hàm ghi log/plot/CSV đều no-op
    monkeypatch.setattr("main.save_run_params", lambda *args, **kwargs: None)
    monkeypatch.setattr("main.save_metrics_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("main.save_metrics_to_csv", lambda *args, **kwargs: None)

    # 2. Chuẩn bị arguments cho main
    args = argparse.Namespace(
        dataset_path=dummy_dataset,
        load_checkpoint_path=None,
        model_type="deeplab",
        batch_size=1,
        epochs=1,
        patience=1,
        backbone=None,
        learning_rate=1e-4,
        seed=42,
        loss_type="combined",
        compile_model=False,
        num_workers=0,
        scheduler_max_lr=None,
        scheduler_pct_start=None,
        scheduler_div_factor=None,
        scheduler_final_div_factor=None,
    )

    # 3. Gọi main() và kiểm tra không ném exception
    try:
        main(args)
    except Exception as e:
        pytest.fail(f"main(args) raised an exception: {e}")
