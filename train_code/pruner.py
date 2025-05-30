import argparse
import os
import torch
import torch.nn.utils.prune as prune_utils
import torch_pruning as tp
from model.deeplab import get_lraspp_model
from config import Config
from train.trainer import validate, train_one_epoch
from torch.utils.data import DataLoader
from data.dataset import LaRSDataset
from utils.losses import get_loss_function
from utils.plotting import save_metrics_plot, save_metrics_to_csv

def apply_unstructured_pruning(model, target_sparsity, layers_to_prune):
    """
    Apply unstructured pruning to the specified layers of the model.
    Args:
        model (torch.nn.Module): The model to prune.
        target_sparsity (float): The target sparsity level (between 0 and 1).
        layers_to_prune (list of tuples): List of tuples containing layer names and modules to prune.
    """
    for name, module in layers_to_prune:
        try:
            prune_utils.l1_unstructured(module, name="weight", amount=target_sparsity)
        except Exception as e:
            print(f"Could not prune {name} in {module}: {e}")
    return model

def make_pruning_permanent(model, layers_to_prune):
    for name, module in layers_to_prune:
        if prune_utils.is_pruned(module):
            prune_utils.remove(module, "weight")
    return model

def get_prunable_layers(model, ignored_modules_list):
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            is_ignored = False
            if ignored_modules_list:
                for ignored_parent_module in ignored_modules_list:
                    if module in ignored_parent_module.modules():
                        is_ignored = True
                        break
            if not is_ignored:
                prunable_layers.append((name, module))
            else:
                print(f"Ignoring pruning for layer: {name}")
    return prunable_layers

def calculate_sparsity(model, layers_to_prune):
    total_params = 0
    zero_params = 0
    for _, module in layers_to_prune:
        if hasattr(module, 'weight'): # Check if weight exists
            total_params += module.weight.nelement()
            zero_params += torch.sum(module.weight == 0).item()
    actual_sparsity = zero_params / total_params if total_params > 0 else 0
    return actual_sparsity

def finetune(
    model,
    config,
    epochs,
    val_loader=None,
    device=None,
    current_epoch_offset=0
):
    """
    Fine-tune the pruned model for a specified number of epochs.
    Returns lists of metrics for this finetuning session.
    """
    model.train()

    train_dataset = LaRSDataset(
        image_dir=config.train_dataset_path,
        image_names=config.train_names,
        mask_dir=config.train_mask_path,
        transform=None, 
        target_size=config.input_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    criterion = get_loss_function(
        config.loss_type, ce_weight=config.ce_weight, dice_weight=config.dice_weight
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate, 
        steps_per_epoch=len(train_loader),
        epochs=epochs, 
        pct_start=0.3, 
        cycle_momentum=False,
    )
    
    session_time_list = []
    session_train_losses = []
    session_val_losses = []
    session_val_accuracies = []
    session_val_mious = []

    for epoch in range(1, epochs + 1):
        actual_epoch_num = current_epoch_offset + epoch
        time_taken, train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            actual_epoch_num, 
            scheduler,
        )
        if val_loader:
            val_accuracy, val_miou, val_loss = validate(
                model, val_loader, criterion, device, config.num_classes, actual_epoch_num
            )
        else:
            val_accuracy, val_miou, val_loss = 0.0, 0.0, float('inf')

        session_time_list.append(time_taken)
        session_train_losses.append(train_loss)
        session_val_losses.append(val_loss)
        session_val_accuracies.append(val_accuracy)
        session_val_mious.append(val_miou)
        
        print(f"Finetune Epoch {epoch}/{epochs} (Global {actual_epoch_num}): Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val mIoU: {val_miou:.4f}")

    print("Fine-tuning complete for this iteration.")
    return session_time_list, session_train_losses, session_val_losses, session_val_accuracies, session_val_mious


def prune_main(args):
    config = Config(dataset_path_override=args.datapath)
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    device_obj = torch.device(config.device)

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found or not specified.")
        return

    model = get_lraspp_model(
        num_classes=config.num_classes,
        device=device_obj)
    for param in model.parameters():
        param.requires_grad = True

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
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device_obj)
    print("Model loaded successfully.")

    example_inputs = torch.randn(1, 3, *config.input_size).to(device_obj)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    val_dataset = LaRSDataset(
        image_dir=config.val_dataset_path,
        image_names=config.val_names,
        mask_dir=config.val_mask_path,
        transform=None,
        target_size=config.input_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
        pin_memory=(device_obj.type == "cuda"),
    )
    criterion = get_loss_function(
        config.loss_type, ce_weight=config.ce_weight, dice_weight=config.dice_weight
    )
    print("Validating model before pruning...")
    val_acc_before, miou_before, val_loss_before = validate(
        model, val_loader, criterion, device_obj, config.num_classes, epoch=0 
    )
    print("Model before pruning:")
    print(
        f"Before pruning: MACs={base_macs / 1e9:.4f}G, Params={base_nparams / 1e6:.4f}M"
    )
    print(
        f"Pixel Accuracy {val_acc_before:.4f},\tmIoU {miou_before:.4f},\tLoss {val_loss_before:.4f}"
    )

    ignored_modules = []
    if hasattr(model, 'classifier'):
        ignored_modules.append(model.classifier)
    
    prunable_layers = get_prunable_layers(model, ignored_modules)
    if not prunable_layers:
        print("No prunable Conv2D layers found (excluding ignored modules). Exiting.")
        return
    print(f"Found {len(prunable_layers)} Conv2D layers to prune.")

    all_iter_times = []
    all_iter_train_losses = []
    all_iter_val_losses = []
    all_iter_val_accuracies = []
    all_iter_val_mious = []
    
    for i in range(args.iterative_steps):
        iteration_num = i + 1
        model.train() 
        print(f"Pruning iteration {iteration_num}/{args.iterative_steps}...")

        current_iteration_target_sparsity = args.target_prune_rate * (iteration_num / args.iterative_steps)
        
        print(f"Applying L1 unstructured pruning to target sparsity: {current_iteration_target_sparsity:.4f}")
        apply_unstructured_pruning(model, current_iteration_target_sparsity, prunable_layers)

        sparsity_after_masking = calculate_sparsity(model, prunable_layers)
        print(f"Sparsity after applying masks (iteration {iteration_num}): {sparsity_after_masking:.4f}")

        # Fine-tune the pruned model
        print(f"Fine-tuning for {args.epochs} epochs (Iteration {iteration_num})...")
        finetune_epoch_offset = i * args.epochs
        (iter_times, iter_train_l, iter_val_l, iter_val_acc, iter_val_miou) = finetune(
            model,
            config,
            args.epochs,
            val_loader=val_loader,
            device=device_obj,
            current_epoch_offset=finetune_epoch_offset,
        )
        all_iter_times.extend(iter_times)
        all_iter_train_losses.extend(iter_train_l)
        all_iter_val_losses.extend(iter_val_l)
        all_iter_val_accuracies.extend(iter_val_acc)
        all_iter_val_mious.extend(iter_val_miou)

        # Validate the pruned model
        print(f"Validating after fine-tuning (Iteration {iteration_num})...")
        val_acc_after_ft, miou_after_ft, val_loss_after_ft = validate(
            model, val_loader, criterion, device_obj, config.num_classes, epoch=finetune_epoch_offset + args.epochs
        )
        print(
            f"After fine-tuning iteration {iteration_num}: Pixel Acc {val_acc_after_ft:.4f}, mIoU {miou_after_ft:.4f}, Loss {val_loss_after_ft:.4f}"
        )

        if miou_after_ft < (1 - args.max_map_drop) * miou_before:
            print(f"mIoU ({miou_after_ft:.4f}) dropped below threshold relative to original mIoU ({miou_before:.4f}) after iteration {iteration_num}. Stopping pruning.")
            break

        # Save the pruned model of each iteration
        original_model_basename = os.path.splitext(os.path.basename(model_path))[0]
        pruned_model_iter_filename = f"pruned_iter_{iteration_num}_sparsity_{current_iteration_target_sparsity:.2f}.pth"
        pruned_model_iter_dir = f"checkpoints/prune/{original_model_basename}"
        os.makedirs(pruned_model_iter_dir, exist_ok=True)
        pruned_model_iter_path = os.path.join(pruned_model_iter_dir, pruned_model_iter_filename)
        
        torch.save(model.state_dict(), pruned_model_iter_path)
        print(f"Pruned model (with masks) for iteration {iteration_num} saved to: {pruned_model_iter_path}")
    
    print("Making pruning permanent on the final model state...")
    make_pruning_permanent(model, prunable_layers)
    
    final_macs, final_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    final_sparsity = calculate_sparsity(model, prunable_layers)
    print(
        f"After making pruning permanent: MACs={final_macs / 1e9:.4f}G, Params={final_nparams / 1e6:.4f}M, Final Sparsity: {final_sparsity:.4f}"
    )

    print("Final validation of the permanently pruned model...")
    val_acc_final, miou_final, val_loss_final = validate(
        model, val_loader, criterion, device_obj, config.num_classes, epoch=-1 
    )
    print(
        f"Permanently pruned model: Pixel Acc {val_acc_final:.4f}, mIoU {miou_final:.4f}, Loss {val_loss_final:.4f}"
    )

    final_pruned_model_filename = f"pruned_final_sparsity_{final_sparsity:.2f}_miou_{miou_final:.3f}.pth"
    final_pruned_model_path = os.path.join(pruned_model_iter_dir, final_pruned_model_filename)
    torch.save(model.state_dict(), final_pruned_model_path)
    print(f"Final permanently pruned model saved to: {final_pruned_model_path}")
    
    model_size_final_mb = os.path.getsize(final_pruned_model_path) / (1024 * 1024)
    print(f"Final pruned model size: {model_size_final_mb:.2f} MB")

    if all_iter_train_losses:
        overall_metrics_filename_stem = f"pruning_summary_{original_model_basename}"
        overall_metrics_csv_path = os.path.join(pruned_model_iter_dir, f"{overall_metrics_filename_stem}.csv")
        overall_plots_path = os.path.join(pruned_model_iter_dir, f"{overall_metrics_filename_stem}.png")
        os.makedirs(os.path.dirname(overall_metrics_csv_path), exist_ok=True)

        save_metrics_plot(
            epochs_range=range(1, len(all_iter_train_losses) + 1),
            train_losses=all_iter_train_losses,
            val_losses=all_iter_val_losses,
            val_accuracies=all_iter_val_accuracies,
            val_mious=all_iter_val_mious,
            plot_path=overall_plots_path,
        )
        save_metrics_to_csv(
            metrics_path=overall_metrics_csv_path,
            time_list=all_iter_times,
            train_losses=all_iter_train_losses,
            val_losses=all_iter_val_losses,
            val_accuracies=all_iter_val_accuracies,
            val_mious=all_iter_val_mious,
            model_size=model_size_final_mb,
        )
        print(f"Overall pruning metrics saved to CSV: {overall_metrics_csv_path}")
        print(f"Overall pruning plots saved to PNG: {overall_plots_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Prune a segmentation model.")
    parser.add_argument("--model", required=True, help="Path to the model to prune")
    parser.add_argument("--datapath", help="Path to the dataset directory")
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
    prune_main(args)


if __name__ == "__main__":
    main()
