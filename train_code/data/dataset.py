import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image

class LaRSDataset(Dataset):
    def __init__(self, image_dir, image_names, mask_dir, transform=None, target_size=(320, 320)):
        self.image_names = image_names
        self.image_paths = image_dir
        self.mask_paths = mask_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_names)
    
    def _load_image_mask_pil(self, image_id):
        """Loads image and mask as PIL objects, resizes, and returns them."""
        image_path = os.path.join(self.image_paths, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_paths, f"{image_id}.png")

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L") # Load as grayscale
        except FileNotFoundError:
            print(f"Warning: File not found: {image_path} or {mask_path}")
            return None, None

        if self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)

        return image, mask

    def __getitem__(self, idx):
        image_id = self.image_names[idx]
        image_pil, mask_pil = self._load_image_mask_pil(image_id)

        if image_pil is None:
            raise FileNotFoundError(f"Image or mask not found for ID {image_id}")

        # Convert PIL images to NumPy array
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil, dtype=np.int64)
        mask_np[mask_np == 255] = 0  # thay ignore index thành background (0)

        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_tensor = augmented['image']       # đã là torch.Tensor
            mask_tensor = augmented['mask'].long()  # đảm bảo dtype = long
        else:
            # Nếu không dùng Albumentations, fallback về ToTensor cho image, mask chỉ convert sang Tensor
            image_tensor = transforms.ToTensor()(image_pil)
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor