from config import Config
from data.dataset import LaRSDataset
from model.deeplab import get_deeplab_model
from train.trainer import train_one_epoch, validate
from utils.plotting import save_metrics_plot, save_metrics_to_csv

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.amp as amp
import os
import random
import numpy as np

config = Config()
run_id = config.run_id  # Lấy mã run_id duy nhất cho lần train này
base_model_name = f"best_model_{config.date_str}_{run_id}"
best_model_path = f"checkpoints/train/{config.date_str}/{base_model_name}.pth"
metrics_path = f"output/train/{config.date_str}/{base_model_name}_metrics.csv"
plots_path = f"output/train/{config.date_str}/{base_model_name}_metrics.png"

# Optional: Add set_seed function here or import from utils if you create one
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = config.cudnn_benchmark
torch.backends.cudnn.deterministic = not config.cudnn_benchmark # Deterministic if benchmark is off

# Load dataset
with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "train", "image_list.txt"), encoding="utf-8") as f:
    train_names = [line.strip() for line in f]
with open(os.path.join(config.dataset_path, "lars_v1.0.0_images", "val", "image_list.txt"), encoding="utf-8") as f:
    val_names = [line.strip() for line in f]

train_dataset = LaRSDataset(
    image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "train", "images"),
    image_names=train_names,
    mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_annotations", "train", "semantic_masks"),
    transform=None, # Add torchvision.transforms.Normalize here if needed (after ToTensor)
    target_size=config.input_size
)

val_dataset = LaRSDataset(
    image_dir=os.path.join(config.dataset_path, "lars_v1.0.0_images", "val", "images"),
    image_names=val_names,
    mask_dir=os.path.join(config.dataset_path, "lars_v1.0.0_annotations", "val", "semantic_masks"),
    transform=None, # Add torchvision.transforms.Normalize here if needed (after ToTensor)
    target_size=config.input_size
)

pin_memory_flag = True if config.device == 'cuda' else False
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=pin_memory_flag)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=pin_memory_flag)

print("Dataset loaded successfully.")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Model, loss function, optimizer
device_obj = torch.device(config.device)
model = get_deeplab_model(num_classes=config.num_classes, device=device_obj)

if hasattr(config, 'load_checkpoint_path') and config.load_checkpoint_path:
    print(f"Loading model from checkpoint: {config.load_checkpoint_path}")
    if os.path.exists(config.load_checkpoint_path):
        model.load_state_dict(torch.load(config.load_checkpoint_path, map_location=device_obj))
        print("Model loaded successfully.")
    else:
        print(f"Checkpoint not found: {config.load_checkpoint_path}. Starting training from scratch.")

if hasattr(torch, 'compile') and device_obj.type == 'cuda':
    print("Attempting to compile the model with torch.compile()...")
    try:
        model = torch.compile(model) # For PyTorch 2.0+
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
scaler = amp.GradScaler(enabled=(device_obj.type == 'cuda'))

# Train loop
train_losses, val_losses, val_accuracies, val_mious = [], [], [], []
best_miou = 0.0
for epoch in range(1, config.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device_obj, scaler, epoch)
    val_accuracy, val_miou, val_loss = validate(model, val_loader, criterion, device_obj, config.num_classes, epoch)
    scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_mious.append(val_miou)

    if val_miou > best_miou:
        best_miou = val_miou
        best_model_dir = os.path.dirname(best_model_path)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir, exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path} with mIoU: {best_miou:.4f}")

# Save training metrics
metrics_dir = os.path.dirname(metrics_path)
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir, exist_ok=True)
save_metrics_plot(
    epochs_range=range(1, config.epochs + 1),
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    val_mious=val_mious,
    plot_path=plots_path
)
save_metrics_to_csv(
    metrics_path=metrics_path,
    train_losses=train_losses,
    val_losses=val_losses,
    val_accuracies=val_accuracies,
    val_mious=val_mious
)