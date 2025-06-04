import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(target_size=(320, 320)):
    """
    Return a Compose of Albumentations to apply on image + mask.
    target_size: (width, height).
    """
    width, height = target_size

    return A.Compose([
        # ----- Resize & giữ tỉ lệ (nếu cần) -----
        # Đảm bảo mask và ảnh luôn về đúng kích thước (320, 320)
        A.Resize(height=height, width=width),

        # ----- Geometric augmentations (đổi vị trí pixel đồng bộ với mask) -----
        # Lật ngang ảnh và mask
        A.HorizontalFlip(p=0.5),
        # Cắt ảnh ngẫu nhiên 
        A.RandomCrop(height=height, width=width, p=0.3),
        # Cắt ảnh ngẫu nhiền và phóng ảnh
        A.RandomResizedCrop(size=(height, width), scale=(0.7, 1.0), ratio=(0.9,1.1), p=0.3),

        # ----- Photometric augmentations (chỉ ảnh, mask không thay đổi nhãn) -----
        # Ánh sáng/chói/overexposure giả lập hiệu ứng thời tiết
        A.RandomSunFlare(src_radius=100, p=0.2),
        A.RandomFog(p=0.2),
        A.RandomRain(p=0.2),
        A.RandomSnow(p=0.1),

        # Bóng tối ngẫu nhiên và mờ viền
        A.RandomShadow(p=0.3),
        # Độ sáng ngãu nhiên
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(p=0.3),

        # Thêm Noise
        A.GaussNoise(p=0.1),
        # Làm mờ
        A.MotionBlur(p=0.1),
        A.GaussianBlur(p=0.1),

        # ----- Chuẩn hóa và convert sang tensor -----
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_augmentations(target_size=(320, 320)):
    """
    Chỉ resize + normalize + toTensor. Không dùng augmentation mạnh mẽ.
    """
    width, height = target_size

    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
