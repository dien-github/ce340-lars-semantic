import os
from torch.utils.data import DataLoader
from data.dataset import LaRSDataset

def load_datasets(config):
    with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "train", "image_list.txt"), encoding="utf-8") as f:
        train_names = [line.strip() for line in f]
    with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "val", "image_list.txt"), encoding="utf-8") as f:
        val_names = [line.strip() for line in f]
    train_dataset = LaRSDataset(
        image_dir=config.train_dataset_path,
        image_names=train_names,
        mask_dir=config.train_mask_path,
        transform=None,  # Add torchvision.transforms.Normalize here if needed (after ToTensor)
        target_size=config.input_size,
    )

    val_dataset = LaRSDataset(
        image_dir=config.val_dataset_path,
        image_names=val_names,
        mask_dir=config.val_mask_path,
        transform=None,  # Add torchvision.transforms.Normalize here if needed (after ToTensor)
        target_size=config.input_size,
    )

    pin_memory_flag = config.device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory_flag,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory_flag,
    )

    print("Dataset loaded successfully.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    return train_dataset, val_dataset, train_loader, val_loader
