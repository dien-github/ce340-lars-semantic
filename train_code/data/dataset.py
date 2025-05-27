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
    
    def __getitem__(self, idx):
        image_id = self.image_names[idx]
        image, mask = self.load_image_mask(image_id)
        if image is None or mask is None:
            raise FileNotFoundError(f"Image or mask not found for ID: {image_id}")
        
        # Convert PIL Images to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask