from torch.utils.data import DataLoader
# from torchvision import transforms # Added for potential normalization
from data.dataset import LaRSDataset
from data.augmentation import get_training_augmentations, get_validation_augmentations

def load_datasets(config):
    train_transforms = get_training_augmentations(target_size=config.input_size)
    val_transforms = get_validation_augmentations(target_size=config.input_size)

    train_dataset = LaRSDataset(
        image_dir=config.train_dataset_path,
        image_names=config.train_names, # Use property from config
        mask_dir=config.train_mask_path,
        transform=train_transforms,
        target_size=config.input_size,
    )

    val_dataset = LaRSDataset(
        image_dir=config.val_dataset_path,
        image_names=config.val_names, # Use property from config
        mask_dir=config.val_mask_path,
        transform=val_transforms, # Apply same non-augmenting transforms to val
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
