from config import Config
from data.dataset import LaRSDataset
from model.deeplab import get_deeplab_model
from train.trainer import train_one_epoch, validate
from utils.plotting import save_metrics_plot, save_metrics_to_csv

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os

config = Config()

# Load dataset
with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "train", "image_list.txt"), encoding="utf-8") as f:
    train_names = [line.strip() for line in f]
with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "val", "image_list.txt"), encoding="utf-8") as f:
    val_names = [line.strip() for line in f]

train_dataset = LaRSDataset(
    image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "train"),
    image_names=train_names,
    mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_masks", "train"),
    transform=None,
    target_size=config.input_size
)

val_dataset = LaRSDataset(
    image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "val"),
    image_names=val_names,
    mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_masks", "val"),
    transform=None,
    target_size=config.input_size
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Model, loss function, optimizer
model = get_deeplab_model(num_classes=config.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Train loop
train_losses, val_losses, val_accuracies, val_mious = [], [], [], []
best_miou = 0.0
for epoch in range(1, config.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device, epoch)
    val_accuracy, val_miou, val_loss = validate(model, val_loader, criterion, config.device, config.num_classes, epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_mious.append(val_miou)

    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), config.best_model_path)
        print(f"Best model saved with mIoU: {best_miou:.4f}")

# Save training metrics
save_metrics_plot(
    epochs_range=range(1, config.epochs + 1),
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    val_mious=val_mious,
    plot_path=config.plots_path
)
save_metrics_to_csv(
    metrics_path=config.metrics_path,
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    val_mious=val_mious
)