from config import Config
from data.load import load_datasets
from model.deeplab import get_deeplab_model, get_lraspp_model
from train.trainer import train_one_epoch, validate
from utils.logging import save_run_params
from utils.losses import get_loss_function
from utils.plotting import save_metrics_plot, save_metrics_to_csv

import argparse
import torch
from torch import optim
import torch.amp as amp
import os

def main(args):
    config = Config()
    #Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    run_id = config.run_id
    base_model_name = f"best_model_{config.date_str}_{run_id}"
    best_model_path = f"checkpoints/train/{config.date_str}/{base_model_name}.pth"
    metrics_path = f"output/train/{config.date_str}/{base_model_name}_metrics.csv"
    plots_path = f"output/train/{config.date_str}/{base_model_name}_metrics.png"

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    # Load dataset
    train_dataset, val_dataset, train_loader, val_loader = load_datasets(config)

    # Model selection
    device_obj = torch.device(config.device)
    model_type = getattr(config, "model_type", "deeplab").lower()
    if model_type == "deeplab":
        model = get_deeplab_model(num_classes=config.num_classes, device=device_obj)
        model_name = "DeepLabV3-MobileNetV3"
    elif model_type == "lraspp":
        model = get_lraspp_model(
            num_classes=config.num_classes,
            device=device_obj,
            freeze_layers=None,
            unfreeze_layers=["backbone", "classifier"],
        )
        model_name = "LRASPP-MobileNetV3"
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in config. Use 'deeplab' or 'lraspp'."
        )

    # Optionally load checkpoint
    if getattr(config, "load_checkpoint_path", None):
        print(f"Loading model from checkpoint: {config.load_checkpoint_path}")
        if os.path.exists(config.load_checkpoint_path):
            # model.load_state_dict(torch.load(config.load_checkpoint_path, map_location=device_obj))
            state_dict = torch.load(config.load_checkpoint_path, map_location=device_obj)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[len("_orig_mod.") :]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully.")
        else:
            print(
                f"Checkpoint not found: {config.load_checkpoint_path}. Starting training from scratch."
            )

    # Optionally compile model (PyTorch 2.0+)
    if hasattr(torch, "compile") and device_obj.type == "cuda":
        print("Attempting to compile the model with torch.compile()...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Proceeding without compilation.")

    criterion = get_loss_function(config.loss_type, ce_weight=config.ce_weight, dice_weight=config.dice_weight)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # StepLR for periodic decay
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=config.learning_rate,
    #     steps_per_epoch=len(train_loader),
    #     epochs=config.epochs,
    #     pct_start=0.1,
    #     anneal_strategy='cos',
    #     cycle_momentum=False
    # )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.3,
        div_factor=100,  # LR start từ 3e-6
        final_div_factor=100,  # kết thúc ~3e-6
        cycle_momentum=False,
    )

    scaler = amp.GradScaler(enabled=(device_obj.type == "cuda"))

    # Training loop
    patience = 7
    epochs_no_improve = 0
    train_losses, val_losses, val_accuracies, val_mious = [], [], [], []
    time_list = []
    best_miou = 0.0

    for epoch in range(1, config.epochs + 1):
        time, train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device_obj,
            scaler,
            epoch,
            scheduler=scheduler,
        )
        val_accuracy, val_miou, val_loss = validate(
            model, val_loader, criterion, device_obj, config.num_classes, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_mious.append(val_miou)
        time_list.append(time)

        if val_miou > best_miou:
            best_miou = val_miou
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved: {best_model_path} with mIoU: {best_miou:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in mIoU for {epochs_no_improve} epochs.")
        print(
            f"Epoch {epoch}/{config.epochs}\n\tTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n\t\tVal Accuracy: {val_accuracy:.4f}, Val mIoU: {val_miou:.4f}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # Size in MB
    print(f"Best model size: {model_size:.2f} MB")

    # Save training metrics and parameters
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    save_metrics_plot(
        epochs_range=range(1, len(train_losses) + 1),
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        val_mious=val_mious,
        plot_path=plots_path,
    )
    save_metrics_to_csv(
        metrics_path=metrics_path,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        val_mious=val_mious,
        model_size=model_size,
        time_list=time_list,
    )

    param_log_path = (
        f"checkpoints/description/{config.date_str}/{base_model_name}_params.txt"
    )
    params = {
        "run_id": run_id,
        "base_model_name": base_model_name,
        "model_type": model_name,
        "num_classes": config.num_classes,
        "input_size": config.input_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "scheduler_params": {
            "max_lr": config.learning_rate,
            "steps_per_epoch": len(train_loader),
            "epochs": config.epochs,
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "cycle_momentum": False,
        },
        "seed": config.seed,
        "dataset_path": config.dataset_path,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "device": config.device,
        "cudnn_benchmark": config.cudnn_benchmark,
        "early_stopping_patience": patience,
        "best_model_path": best_model_path,
        "metrics_path": metrics_path,
        "plots_path": plots_path,
    }
    save_run_params(param_log_path, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to a checkpoint to load the model from')
    parser.add_argument('-m', '--model_type', type=str, choices=['deeplab', 'lraspp'], help='Type of model to use')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--backbone', type=str, help='Backbone model to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'dice', 'combined'], help='Type of loss function to use')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading.')
    args = parser.parse_args()
    main(args)