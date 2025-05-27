import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import zipfile

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed=None, seed_torch=True):
    """
    Set random seed for reproducibility.

    Args:
        seed (int, optional): Random seed. If None, a random seed is chosen.
        seed_torch (bool): If True, also set PyTorch seeds.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")

def load_image_mask(image_id, image_dir, mask_dir, target_size=(320, 320)):
    """
    Load an image and its corresponding mask from directories.

    Args:
        image_id (str): Image/mask identifier (without extension).
        image_dir (str): Directory with images.
        mask_dir (str): Directory with masks.
        target_size (tuple): Resize (width, height).

    Returns:
        tuple: (PIL.Image image, PIL.Image mask)
    """
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    mask_path = os.path.join(mask_dir, f"{image_id}.png")

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if target_size:
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

    return image, mask

def train_model(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Train] Epoch {epoch}")
    for batch_idx, (images, masks) in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_model(model, dataloader, criterion, device, epoch=None):
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    iou_scores = [[] for _ in range(NUM_CLASSES)]
    running_val_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), desc=f"[Val] Epoch {epoch}")

    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, masks)
            running_val_loss += loss.item()
            total_pixels += masks.numel()
            correct_pixels += (predicted == masks).sum().item()

            for cls in range(NUM_CLASSES):
                intersection = ((predicted == cls) & (masks == cls)).sum().item()
                union = ((predicted == cls) | (masks == cls)).sum().item()
                if union > 0:
                    iou_scores[cls].append(intersection / union)

            loop.set_postfix(iou_scores=[np.mean(iou) if iou else 0 for iou in iou_scores])
    mean_iou = np.mean([np.mean(iou) if iou else 0 for iou in iou_scores])
    accuracy = correct_pixels / total_pixels
    epoch_val_loss = running_val_loss / len(dataloader.dataset)
    print(f"Validation Accuracy: {accuracy:.4f}, Loss: {epoch_val_loss:.4f}, mIoU: {mean_iou:.4f}")
    return accuracy, mean_iou, epoch_val_loss


# Define the dataset class
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
        image, mask = load_image_mask(image_id, self.image_paths, self.mask_paths, self.target_size)
        if image is None or mask is None:
            raise FileNotFoundError(f"Image or mask not found for ID: {image_id}")
        if self.transform and not isinstance(self.transform, transforms.ToTensor):
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        # Replace 255 (ignore index) with 0 (background) or another valid class index
        mask[mask == 255] = 0
        return image, mask



if __name__ == "__main__":
    set_seed(seed=2025)
    NUM_CLASSES = 4
    BATCH_SIZE = 64
    EPOCHS = 10

    # Set up paths and load dataset
    dataset_path = os.path.join("ce340-lars-semantic", "LaRS_dataset")

    train_images_path = os.path.join(dataset_path, "lars_v1.0.0_images", "train", "images")
    train_masks_path = os.path.join(dataset_path, "lars_v1.0.0_annotations", "train", "semantic_masks")
    with open(os.path.join(dataset_path, "lars_v1.0.0_images", "train", "image_list.txt"), encoding="utf-8") as f:
        train_names = [line.strip() for line in f]

    val_images_path = os.path.join(dataset_path, "lars_v1.0.0_images", "val", "images")
    val_masks_path = os.path.join(dataset_path, "lars_v1.0.0_annotations", "val", "semantic_masks")
    with open(os.path.join(dataset_path, "lars_v1.0.0_images", "val", "image_list.txt"), encoding="utf-8") as f:
        val_names = [line.strip() for line in f]

    train_dataset = LaRSDataset(
        image_dir=train_images_path,
        image_names=train_names,
        mask_dir=train_masks_path,
        transform=transforms.ToTensor(),
        target_size=(320, 320)
    )
    val_dataset = LaRSDataset(
        image_dir=val_images_path,
        image_names=val_names,
        mask_dir=val_masks_path,
        transform=transforms.ToTensor(),
        target_size=(320, 320)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("Dataset loaded successfully.")
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))

    # Load the pre-trained DeepLabV3 model with MobileNetV3 backbone
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        NUM_CLASSES=NUM_CLASSES,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_mious = []
    best_miou = 0.0
    best_model_path = "best_model.pth"
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
        pixel_acc, miou, val_loss = validate_model(model, val_loader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(pixel_acc)
        val_mious.append(miou)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with mIoU: {best_miou:.4f}")

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs_range, val_losses, label="Val Loss", color='orange')
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Pixel Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label="Pixel Acc", color='green')
    plt.title("Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # mIoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_mious, label="mIoU", color='red')
    plt.title("Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

    # Save training metrics to CSV
    with open("training_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Pixel Accuracy", "Mean IoU"])
        for i in range(EPOCHS):
            writer.writerow([i+1, train_losses[i], val_losses[i], val_accuracies[i], val_mious[i]])

# from config import Config
# from data.dataset import LaRSDataset
# from model.deeplab import get_deeplab_model
# from train.trainer import train_one_epoch, validate
# from utils.plotting import save_metrics_plot, save_metrics_to_csv

# import torch
# from torch.utils.data import DataLoader
# from torch import nn, optim
# import os

# config = Config()

# # Load dataset
# with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "train", "image_list.txt"), encoding="utf-8") as f:
#     train_names = [line.strip() for line in f]
# with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "val", "image_list.txt"), encoding="utf-8") as f:
#     val_names = [line.strip() for line in f]

# train_dataset = LaRSDataset(
#     image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "train"),
#     image_names=train_names,
#     mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_masks", "train"),
#     transform=None,
#     target_size=config.input_size
# )

# val_dataset = LaRSDataset(
#     image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "val"),
#     image_names=val_names,
#     mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_masks", "val"),
#     transform=None,
#     target_size=config.input_size
# )

# train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# # Model, loss function, optimizer
# model = get_deeplab_model(num_classes=config.num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# # Train loop
# train_losses, val_losses, val_accuracies, val_mious = [], [], [], []
# best_miou = 0.0
# for epoch in range(1, config.epochs + 1):
#     train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device, epoch)
#     val_accuracy, val_miou, val_loss = validate(model, val_loader, criterion, config.device, config.num_classes, epoch)

#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
#     val_mious.append(val_miou)

#     if val_miou > best_miou:
#         best_miou = val_miou
#         torch.save(model.state_dict(), config.best_model_path)
#         print(f"Best model saved with mIoU: {best_miou:.4f}")

# # Save training metrics
# save_metrics_plot(
#     epochs_range=range(1, config.epochs + 1),
#     train_losses=train_losses,
#     val_losses=val_losses,
#     val_accuracies=val_accuracies,
#     val_mious=val_mious,
#     plot_path=config.plots_path
# )
# save_metrics_to_csv(
#     metrics_path=config.metrics_path,
#     train_losses=train_losses,
#     val_losses=val_losses,
#     val_accuracies=val_accuracies,
#     val_mious=val_mious
# )