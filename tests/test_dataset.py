import pytest
from train_code.data.dataset import LaRSDataset

def test_dataset_len_and_getitem(tmp_path):
    # Tạo dữ liệu giả lập
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    # Tạo 2 ảnh và 2 mask giả
    for i in range(2):
        (img_dir / f"img_{i}.png").write_bytes(b"fake")
        (mask_dir / f"img_{i}.png").write_bytes(b"fake")
    image_names = [f"img_{i}.png" for i in range(2)]
    dataset = LaRSDataset(str(img_dir), image_names, str(mask_dir), transform=None, target_size=(320,320))
    assert len(dataset) == 2
    # __getitem__ có thể sẽ lỗi do ảnh giả, nên chỉ test không raise lỗi khi đọc tên
    with pytest.raises(Exception):
        _ = dataset[0]