from config import Config
from data.load import load_datasets
from model.deeplab import get_deeplab_model, get_lraspp_model
from model.fscnn_mobilenetv3 import get_fscnn_mobilenetv3_model
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
    # Initialize Config with the dataset_path from arguments
    # This ensures all paths within Config are correctly based on the runtime environment (e.g., Kaggle)
    config = Config(dataset_path_override=args.dataset_path)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            # print(f"Updating config: {key} = {value}") # dataset_path is already handled by __init__
            setattr(config, key, value)

    # Paths are now properties of config and will use the calculated run_id and base_model_name
    # best_model_path = config.best_model_path (example, used directly later)
    # metrics_path = config.metrics_path (example, used directly later)
    # plots_path = config.plots_path (example, used directly later)

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
    # ADD THIS NEW ELIF BLOCK
    elif model_type == "fscnn_mobilenetv3":
        model = get_fscnn_mobilenetv3_model(
            num_classes=config.num_classes,
            device=device_obj,
            pretrained_backbone=True,
            freeze_layers=["backbone"],  # Freeze backbone
            unfreeze_layers=[
                "classifier",
                "conv_low_level",
                "concat_conv",
            ],  # Fine-tune head
        )
        model_name = "FSCNN-MobileNetV3"
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in config. Use 'deeplab', 'lraspp', or 'fscnn_mobilenetv3'."
        )

    # Optionally load checkpoint
    if getattr(config, "load_checkpoint_path", None):
        print(f"Loading model from checkpoint: {config.load_checkpoint_path}")
        if os.path.exists(config.load_checkpoint_path):
            # model.load_state_dict(torch.load(config.load_checkpoint_path, map_location=device_obj))
            state_dict = torch.load(
                config.load_checkpoint_path, map_location=device_obj
            )
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

    # Check for multiple GPUs and use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(
        device_obj
    )  # Ensure model is on the correct device after potentially wrapping with DataParallel

    # Optionally compile model (PyTorch 2.0+)
    # Note: torch.compile() with DataParallel might have some issues or specific behaviors.
    # It's often recommended to use DistributedDataParallel for more complex scenarios with torch.compile.
    # For simplicity with DataParallel, you might skip compilation or test thoroughly.
    # If you encounter issues, consider disabling compile when using DataParallel for now.
    if args.compile_model and hasattr(torch, "compile") and device_obj.type == "cuda":
        if isinstance(model, torch.nn.DataParallel):
            print(
                "torch.compile() with nn.DataParallel might have limitations. Proceeding with caution."
            )
        # print("Attempting to compile the model with torch.compile()...")
        # try:
        #     model = torch.compile(model) # Compiling DataParallel(model)
        #     print("Model compiled successfully.")
        # except Exception as e:
        #     print(f"Model compilation failed: {e}. Proceeding without compilation.")
        print(
            "Skipping torch.compile() when using nn.DataParallel for wider compatibility for now."
        )

    criterion = get_loss_function(
        config.loss_type,
        num_classes=config.num_classes,
        ce_weight=config.ce_weight,
        dice_weight=config.dice_weight,
        lap_weight=config.lap_weight,
    )
    criterion.to(device_obj)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # StepLR for periodic decay
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )

    # Configure OneCycleLR using values from config or provide specific ones in Config class
    # Ensure scheduler_epochs matches the actual number of training epochs
    scheduler_max_lr = getattr(config, "scheduler_max_lr", config.learning_rate)
    scheduler_epochs = (
        config.epochs
    )  # Use config.epochs which might have been updated by args
    scheduler_pct_start = getattr(config, "scheduler_pct_start", 0.3)
    scheduler_div_factor = getattr(config, "scheduler_div_factor", 100.0)
    scheduler_final_div_factor = getattr(config, "scheduler_final_div_factor", 100.0)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scheduler_max_lr,
        steps_per_epoch=len(train_loader),
        epochs=scheduler_epochs,
        pct_start=scheduler_pct_start,
        div_factor=scheduler_div_factor,
        final_div_factor=scheduler_final_div_factor,
        cycle_momentum=False,
    )

    scaler = amp.GradScaler(enabled=(device_obj.type == "cuda"))

    # Training loop
    patience = config.patience
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
            os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
            # If using DataParallel, save the underlying model's state_dict
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), config.best_model_path)
            else:
                torch.save(model.state_dict(), config.best_model_path)
            print(
                f"Best model saved: {config.best_model_path} with mIoU: {best_miou:.4f}"
            )
        else:
            epochs_no_improve += 1
            print(f"No improvement in mIoU for {epochs_no_improve} epochs.")
        print(
            f"Epoch {epoch}/{config.epochs}\n\tTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n\t\tVal Accuracy: {val_accuracy:.4f}, Val mIoU: {val_miou:.4f}"
        )

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    # Ensure the best model was actually saved before trying to get its size
    if os.path.exists(config.best_model_path):
        model_size = os.path.getsize(config.best_model_path) / (
            1024 * 1024
        )  # Size in MB
    else:
        print(
            f"Warning: Best model path {config.best_model_path} not found. Size will be 0."
        )
        model_size = 0
    print(f"Best model size: {model_size:.2f} MB")

    # Save training metrics and parameters
    os.makedirs(os.path.dirname(config.metrics_path), exist_ok=True)
    save_metrics_plot(
        epochs_range=range(1, len(train_losses) + 1),
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        val_mious=val_mious,
        plot_path=config.plots_path,
    )
    save_metrics_to_csv(
        metrics_path=config.metrics_path,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        val_mious=val_mious,
        model_size=model_size,
        time_list=time_list,
    )

    param_log_path = (
        f"checkpoints/description/{config.date_str}/{config.base_model_name}_params.txt"
    )
    os.makedirs(os.path.dirname(param_log_path), exist_ok=True)

    params = {
        "run_id": config.run_id,  # Access property to ensure it's calculated
        "base_model_name": config.base_model_name,  # Access property
        "model_type": model_name,
        "num_classes": config.num_classes,
        "input_size": config.input_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "scheduler_params": {  # Log the actual scheduler parameters used
            "max_lr": scheduler_max_lr,
            "steps_per_epoch": len(train_loader),
            "epochs": scheduler_epochs,
            "pct_start": scheduler_pct_start,
            "div_factor": scheduler_div_factor,
            "final_div_factor": scheduler_final_div_factor,
            "cycle_momentum": False,
        },
        "seed": config.seed,
        "dataset_path": config.dataset_path,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "device": config.device,
        "cudnn_benchmark": config.cudnn_benchmark,
        "early_stopping_patience": patience,
        "best_model_path": config.best_model_path,
        "metrics_path": config.metrics_path,
        "plots_path": config.plots_path,
    }
    save_run_params(param_log_path, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint to load the model from",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        choices=["deeplab", "lraspp", "fscnn_mobilenetv3"],
        help="Type of model to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--backbone", type=str, help="Backbone model to use")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["cross_entropy", "dice", "combined"],
        help="Type of loss function to use",
    )
    # Add an argument for model compilation if you want to control it
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Enable torch.compile for the model",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience for validation loss improvement.",
    )
    args = parser.parse_args()

    main(args)
