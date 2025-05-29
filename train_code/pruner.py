import argparse
import math
import os
import torch
import torch_pruning as tp
from model.deeplab import get_lraspp_model
from config import Config
from train.trainer import validate, train_one_epoch
from torch.utils.data import DataLoader
from data.dataset import LaRSDataset
from utils.losses import get_loss_function
from utils.plotting import save_metrics_plot, save_metrics_to_csv


def finetune(
    model,
    config,
    epochs,
    val_loader=None,
    time_list=None,
    train_losses=None,
    val_losses=None,
    val_accuracies=None,
    val_mious=None,
    device=None
):
    """
    Fine-tune the pruned model for a specified number of epochs.
    """
    model.train()

    train_dataset = LaRSDataset(
        image_dir=config.train_dataset_path,
        image_names=config.train_names,
        mask_dir=config.train_mask_path,
        transform=None,  # Add torchvision.transforms.Normalize here if needed (after ToTensor)
        target_size=config.input_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = get_loss_function(
        config.loss_type, ce_weight=config.ce_weight, dice_weight=config.dice_weight
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.3,
        div_factor=100,  # LR start từ 3e-6
        final_div_factor=100,  # kết thúc ~3e-6
        cycle_momentum=False,
    )

    for epoch in range(epochs):
        time, train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            scheduler,
        )
        val_accuracy, val_miou, val_loss = validate(
            model, val_loader, criterion, device, config.num_classes, epoch
        )
        metrics = [
            (time_list, time),
            (train_losses, train_loss),
            (train_losses, train_loss),
            (val_losses, val_loss),
            (val_accuracies, val_accuracy),
            (val_mious, val_miou),
        ]
        for metric_list, value in metrics:
            if metric_list is not None:
                metric_list.append(value)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")

    print("Fine-tuning complete.")


def prune(args):
    config = Config()
    device_obj = torch.device(config.device)

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found or not specified.")
        return

    model = get_lraspp_model(
        num_classes=config.num_classes,
        device=device_obj,
        freeze_layers=config.freeze_layers,
        unfreeze_layers=config.unfreeze_layers,
    )

    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device_obj)
    # model.load_state_dict(state_dict)
    # state_dict = torch.load(config.load_checkpoint_path, map_location=device_obj)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    model.to(device_obj)
    print("Model loaded successfully.")

    example_inputs = torch.randn(1, 3, *config.input_size)
    example_inputs.to(device_obj)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    val_dataset = LaRSDataset(
        image_dir=config.val_dataset_path,
        image_names=config.val_names,
        mask_dir=os.path.join(
            config.dataset_path, "lars_v1.0.0_annotations", "val", "semantic_masks"
        ),
        transform=None,  # Add torchvision.transforms.Normalize here if needed (after ToTensor)
        target_size=config.input_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device_obj == "cuda"),
    )
    criterion = get_loss_function(
        config.loss_type, ce_weight=config.ce_weight, dice_weight=config.dice_weight
    )
    metric = validate(
        model, val_loader, criterion, device_obj, config.num_classes, epoch=0
    )

    print("Model before pruning:")
    print(
        f"Before pruning: MACs={base_macs / 1e9:.4f}G, Params={base_nparams / 1e6:.4f}M"
    )
    print(metric)

    pruning_ratio = 1 - math.pow((1 - args.prune_rate), 1 / args.iterative_steps)
    print(f"Pruning ratio: {pruning_ratio:.4f}")
    for i in range(args.iterative_steps):
        model.train()
        print(f"Pruning iteration {i + 1}/{args.iterative_steps}...")
        ignored_layers = [model.classifier]
        example_inputs = example_inputs.to(device_obj)
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(
            f"After pruning iteration {i + 1}: MACs={base_macs / 1e9:.4f}G, Params={base_nparams / 1e6:.4f}M"
        )

        # Fine-tune the pruned model
        print(f"Fine-tuning for {args.epochs} epochs...")
        train_losses, val_losses, val_accuracies, val_mious = [], [], [], []
        time_list = []
        finetune(
            model,
            config,
            args.epochs,
            val_loader=val_loader,
            time_list=time_list,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            val_mious=val_mious,
            device=device_obj,
        )

        # Validate the pruned model
        metric = validate(
            model, val_loader, device_obj, config.num_classes, epoch=i + 1
        )
        print(f"After pruning iteration {i + 1}: {metric}")
        initial_miou = metric["mIoU"]
        if metric["mIoU"] < (1 - args.max_map_drop) * initial_miou:
            print(f"mIoU drop exceeded after iteration {i + 1}. Stopping pruning.")
            break

        # Save the pruned model of each iteration
        original_model_basename = os.path.splitext(os.path.basename(model_path))[0]
        pruned_model_save_path = config.pruned_model_path(
            base_model_name=original_model_basename, iteration=i + 1
        )
        os.makedirs(os.path.dirname(pruned_model_save_path), exist_ok=True)
        torch.save(model.state_dict(), pruned_model_save_path)
        # print(f"Pruned model saved to: {pruned_model_save_path}")

        # Print model size before and after pruning
        print(
            f"Original model size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB"
        )
        model_size = os.path.getsize(pruned_model_save_path)
        print(f"Pruned model size: {model_size / (1024 * 1024):.2f} MB")
        # print(f"Pruned model size: {os.path.getsize(pruned_model_save_path) / (1024 * 1024):.2f} MB"        )

        # Save pruning metrics and parameters
        metrics_path = (
            f"checkpoints/prune/{original_model_basename}/metrics_iteration_{i + 1}.csv"
        )
        plots_path = (
            f"checkpoints/prune/{original_model_basename}/plots_iteration_{i + 1}.png"
        )
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
            time_list=time_list,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            val_mious=val_mious,
            model_size=model_size / (1024 * 1024),  # Convert to MB
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Prune a segmentation model.")
    parser.add_argument("--model", required=True, help="Path to the model to prune")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to finetune")
    parser.add_argument("--prune_rate", type=float, default=0.5, help="Pruning ratio")
    parser.add_argument(
        "--iterative_steps", type=int, default=1, help="Total pruning iteration steps"
    )
    parser.add_argument(
        "--target_prune_rate", type=float, default=0.4, help="Target pruning rate"
    )
    parser.add_argument(
        "--max_map_drop",
        type=float,
        default=0.2,
        help="Allowed maximum mIoU drop after fine-tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prune(args)


if __name__ == "__main__":
    main()
