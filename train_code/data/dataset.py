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
            # Consider logging this error or handling it more gracefully
            print(f"Warning: File not found for image_id {image_id} at {image_path} or {mask_path}")
            return None, None

        if self.target_size:
            image = image.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)
        
        return image, mask

    def __getitem__(self, idx):
        image_id = self.image_names[idx]
        image_pil, mask_pil = self._load_image_mask_pil(image_id)

        if image_pil is None or mask_pil is None:
            raise FileNotFoundError(f"Image or mask not found for ID: {image_id}")
        
        # Apply transformations to image (e.g., ToTensor, Normalize)
        image_tensor = transforms.ToTensor()(image_pil) # ToTensor should be applied first to PIL image
        if self.transform: # self.transform should expect a tensor if ToTensor is applied before
            image_tensor = self.transform(image_tensor)
        
        # Process mask: convert to numpy, handle ignore_index, then convert to tensor
        mask_numpy = np.array(mask_pil, dtype=np.int64)
        mask_numpy[mask_numpy == 255] = 0  # Replace ignore index (255) with a valid class (e.g., 0 for background)
        mask_tensor = torch.from_numpy(mask_numpy)
        
        return image_tensor, mask_tensor