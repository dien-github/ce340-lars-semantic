import argparse
import os
import torch
import torch_pruning as tp
from model.deeplab import get_lraspp_model
from config import Config
from train.trainer import validate, train_one_epoch
from torch.utils.data import DataLoader
from data.augmentation import get_training_augmentations, get_validation_augmentations
from data.dataset import LaRSDataset
from utils.losses import get_loss_function
from utils.plotting import save_metrics_plot, save_metrics_to_csv


def apply_structured_pruning_step(
    model: torch.nn.Module,
    channel_pruning_ratio: float,
    example_inputs: torch.Tensor,
    device: torch.device,
    ignored_layers: list = None,
    importance_measure: str = "L1",
) -> torch.nn.Module:
    """
    Apply one step of structured pruning to the model.
    """
    if not (0 < channel_pruning_ratio < 1):
        print("channel_pruning_ratio is out of (0,1), skipping pruning step.")
        return model

    model_cpu = model.cpu()
    example_inputs_cpu = example_inputs.cpu()

    if importance_measure == "L1":
        imp = tp.importance.MagnitudeImportance(p=1)
    elif importance_measure == "L2":
        imp = tp.importance.MagnitudeImportance(p=2)
    elif importance_measure == "Random":
        imp = tp.importance.RandomImportance()
    else:
        raise ValueError(f"Unsupported importance_measure: {importance_measure}")

    # Create a MagnitudePruner with ch_sparsity = channel_pruning_ratio
    pruner = tp.pruner.MagnitudePruner(
        model=model_cpu,
        example_inputs=example_inputs_cpu,
        importance=imp,
        pruning_ratio=channel_pruning_ratio,  # Sửa lại tên tham số
        ignored_layers=ignored_layers or [],
    )

    print(
        f"Applying structured pruning: global filter ratio = {channel_pruning_ratio:.4f}"
    )
    pruner.step()
    return model_cpu.to(device)


def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """
    Remove pruning hooks and clear CUDA cache.
    """
    model_cpu = model.cpu()
    torch.cuda.empty_cache()
    return model_cpu


def collect_ignored_conv_layers(
    model: torch.nn.Module, ignored_parent_modules: list
) -> list:
    """
    Return a list of Conv2d instances under the given parent modules.
    """
    ignored = []
    for parent in ignored_parent_modules:
        for m in parent.modules():
            if isinstance(m, torch.nn.Conv2d):
                ignored.append(m)
    return ignored


def finetune(
    model: torch.nn.Module,
    config: Config,
    epochs: int,
    val_loader: DataLoader = None,
    device: torch.device = None,
    current_epoch_offset: int = 0,
    best_model_save_path: str = None,  # New parameter
):
    """
    Fine-tune the pruned model for a given number of epochs.
    Returns metric lists for this session.
    """
    model = model.to(device)
    model.train()

    train_transform = get_training_augmentations(target_size=config.input_size)
    train_dataset = LaRSDataset(
        image_dir=config.train_dataset_path,
        image_names=config.train_names,
        mask_dir=config.train_mask_path,
        transform=train_transform,
        target_size=config.input_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    criterion = get_loss_function(
        config.loss_type,
        num_classes=config.num_classes,
        ce_weight=config.ce_weight,
        dice_weight=config.dice_weight,
        lap_weight=config.lap_weight,
    ).to(device)

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

    best_miou_finetune = -1.0  # Initialize with a value lower than any possible mIoU
    epochs_no_improve = 0
    patience = getattr(config, 'patience', 10)  # Get patience from config, default to 10
    path_to_best_model_saved_this_finetune = None

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
        if val_loader is not None:
            val_accuracy, val_miou, val_loss = validate(
                model, val_loader, criterion, device, config.num_classes, actual_epoch_num
            )
        else:
            val_accuracy, val_miou, val_loss = 0.0, 0.0, float("inf")

        session_time_list.append(time_taken)
        session_train_losses.append(train_loss)
        session_val_losses.append(val_loss)
        session_val_accuracies.append(val_accuracy)
        session_val_mious.append(val_miou)

        print(
            f"[Fine-tune] Epoch {epoch}/{epochs} (Global {actual_epoch_num}): "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Val Acc={val_accuracy:.4f}, Val mIoU={val_miou:.4f}"
        )

        # Early Stopping Logic & Save Best Model for this finetune session
        if val_miou > best_miou_finetune:
            best_miou_finetune = val_miou
            epochs_no_improve = 0
            if best_model_save_path:
                os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), best_model_save_path)
                else:
                    torch.save(model.state_dict(), best_model_save_path)
                path_to_best_model_saved_this_finetune = best_model_save_path
                print(f"Saved best model for this finetune session at {best_model_save_path} with mIoU: {best_miou_finetune:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in mIoU for {epochs_no_improve} epochs during finetune.")
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement in finetune session.")
            break # Exit the finetune epoch loop

    print("Fine-tuning for this iteration completed.")
    return (
        session_time_list,
        session_train_losses,
        session_val_losses,
        session_val_accuracies,
        session_val_mious,
        path_to_best_model_saved_this_finetune
    )


def prune_main(args):
    config = Config(dataset_path_override=args.datapath)
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    device = torch.device(config.device)

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        print(f"[Error] Model path '{model_path}' not found or not specified.")
        return

    model = get_lraspp_model(num_classes=config.num_classes, device=device)
    for p in model.parameters():
        p.requires_grad = True

    print(f"Loading model from: {model_path} ...")
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"[Error] Unable to load state_dict: {e}")
        return

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warning] Missing keys when loading model: {missing}")
        print(f"[Warning] Unexpected keys: {unexpected}")
    print("Model loaded successfully.")
    model = model.to(device)

    example_inputs = torch.randn(1, 3, *config.input_size).to(device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    val_transform = get_validation_augmentations(target_size=config.input_size)
    val_dataset = LaRSDataset(
        image_dir=config.val_dataset_path,
        image_names=config.val_names,
        mask_dir=config.val_mask_path,
        transform=val_transform,
        target_size=config.input_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    criterion = get_loss_function(
        config.loss_type,
        num_classes=config.num_classes,
        ce_weight=config.ce_weight,
        dice_weight=config.dice_weight,
        lap_weight=config.lap_weight,
    ).to(device)

    print("Validating model before pruning ...")
    val_acc_before, miou_before, val_loss_before = validate(
        model, val_loader, criterion, device, config.num_classes, epoch=0
    )
    print("=== Before pruning ===")
    print(f"MACs={base_macs / 1e9:.4f}G, Params={base_nparams / 1e6:.4f}M")
    print(
        f"Pixel Acc={val_acc_before:.4f}, mIoU={miou_before:.4f}, Loss={val_loss_before:.4f}"
    )

    ignored_parent_modules = []
    if hasattr(model, "classifier"):
        ignored_parent_modules.append(model.classifier)
    ignored_layers = collect_ignored_conv_layers(model, ignored_parent_modules)

    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module not in ignored_layers:
            prunable_layers.append((name, module))
        elif isinstance(module, torch.nn.Conv2d) and module in ignored_layers:
            print(f"Ignoring layer for prune: {name}")
    if not prunable_layers:
        print(
            "[Info] No Conv2d layers found for pruning (after excluding ignored). Exiting."
        )
        return
    print(f"Found {len(prunable_layers)} Conv2d layers to prune.")

    R_total = args.target_prune_rate
    N_steps = args.iterative_steps
    if N_steps > 0 and 0 < R_total < 1:
        r_step = 1.0 - (1.0 - R_total) ** (1.0 / N_steps)
    else:
        r_step = 0.0
    print(f"Target overall prune rate: {R_total:.4f}")
    print(f"Iterative steps: {N_steps}")
    print(f"→ Prune rate per step: {r_step:.4f}")

    all_times = []
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_val_mious = []

    original_basename = os.path.splitext(os.path.basename(model_path))[0]
    base_save_dir = os.path.join("checkpoints", "prune_structured", original_basename)
    os.makedirs(base_save_dir, exist_ok=True)

    model = model.to(device)
    for i in range(N_steps):
        iteration_num = i + 1
        print(f"\n========== Pruning iteration {iteration_num}/{N_steps} ==========")
        model.train()

        # Define save path for the best model of this finetuning iteration (model with masks)
        iter_save_dir = os.path.join(base_save_dir, f"iter_{iteration_num}")
        os.makedirs(iter_save_dir, exist_ok=True) # Ensure iter_save_dir exists before using it
        best_model_finetune_iter_filename = f"best_finetune_iter{iteration_num}.pth" # Simplified name for now
        best_model_finetune_iter_path = os.path.join(iter_save_dir, best_model_finetune_iter_filename)

        model = apply_structured_pruning_step(
            model=model,
            channel_pruning_ratio=r_step,
            example_inputs=example_inputs,
            device=device,
            ignored_layers=ignored_layers,
            importance_measure="L1",
        )

        current_macs, current_params = tp.utils.count_ops_and_params(
            model, example_inputs
        )
        reduced_params = (
            (base_nparams - current_params) / base_nparams if base_nparams > 0 else 0.0
        )
        reduced_macs = (base_macs - current_macs) / base_macs if base_macs > 0 else 0.0
        print(
            f"After prune {iteration_num}: MACs={current_macs / 1e9:.4f}G "
            f"(-{reduced_macs * 100:.2f}%), Params={current_params / 1e6:.4f}M "
            f"(-{reduced_params * 100:.2f}%)"
        )

        print(f"Fine-tuning for {args.epochs} epochs (Iteration {iteration_num}) ...")
        finetune_epoch_offset = i * args.epochs
        (
            iter_times,
            iter_train_losses,
            iter_val_losses,
            iter_val_accuracies,
            iter_val_mious,
            path_to_best_model_this_iter_finetune,
        ) = finetune(
            model=model,
            config=config,
            epochs=args.epochs,
            val_loader=val_loader,
            device=device,
            current_epoch_offset=finetune_epoch_offset,
            best_model_save_path=best_model_finetune_iter_path )
        all_times.extend(iter_times)
        all_train_losses.extend(iter_train_losses)
        all_val_losses.extend(iter_val_losses)
        all_val_accuracies.extend(iter_val_accuracies)
        all_val_mious.extend(iter_val_mious)

        print(f"Validating after fine-tune iteration {iteration_num} ...")
        # Load the best model from this finetuning session before validation and next step
        if path_to_best_model_this_iter_finetune and os.path.exists(path_to_best_model_this_iter_finetune):
            print(f"Loading best model from finetune iteration {iteration_num}: {path_to_best_model_this_iter_finetune}")
            best_state_dict_iter = torch.load(path_to_best_model_this_iter_finetune, map_location=device)
            model.load_state_dict(best_state_dict_iter)
            model.to(device) # Ensure model is on correct device
        else:
            print(f"No best model saved during finetune for iteration {iteration_num}. Using model state at end of finetuning.")
        val_acc_after, miou_after, val_loss_after = validate(
            model,
            val_loader,
            criterion,
            device,
            config.num_classes,
            epoch=finetune_epoch_offset + len(iter_train_losses), # Use actual epochs run
        )
        print(
            f"[Iteration {iteration_num}] Pixel Acc={val_acc_after:.4f}, "
            f"mIoU={miou_after:.4f}, Loss={val_loss_after:.4f}"
        )

        if miou_after < (1 - args.max_map_drop) * miou_before:
            print(
                f"[Warning] mIoU ({miou_after:.4f}) dropped below threshold compared to original ({miou_before:.4f}). "
                f"Stopping at iteration {iteration_num}."
            )
            break

        # The best model from finetune (path_to_best_model_this_iter_finetune) is already saved.
        # We might want to rename it or save another copy with more pruning info if needed,
        # but for now, we'll use the one saved by finetune.
        # The model state in `model` variable is now the best from finetune.
        pruned_model_name = (
            f"pruned_iter{iteration_num}_paramRed_{reduced_params:.2f}.pth"
        )
        pruned_model_path = os.path.join(iter_save_dir, pruned_model_name)
        torch.save(model.state_dict(), pruned_model_path)
        print(
            f"Saved (best from finetune) model for iteration {iteration_num} at: {pruned_model_path}"
        )

        metrics_stem = f"metrics_iter{iteration_num}_paramRed_{reduced_params:.2f}"
        csv_path = os.path.join(iter_save_dir, f"{metrics_stem}.csv")
        plot_path = os.path.join(iter_save_dir, f"{metrics_stem}.png")
        save_metrics_plot(
            epochs_range=range(1, len(iter_train_losses) + 1),
            train_losses=iter_train_losses,
            val_losses=iter_val_losses,
            val_accuracies=iter_val_accuracies,
            val_mious=iter_val_mious,
            plot_path=plot_path,
        )
        save_metrics_to_csv(
            metrics_path=csv_path,
            time_list=iter_times,
            train_losses=iter_train_losses,
            val_losses=iter_val_losses,
            val_accuracies=iter_val_accuracies,
            val_mious=iter_val_mious,
            model_size=os.path.getsize(pruned_model_path) / (1024 * 1024),
        )
        print(
            f"Metrics for iteration {iteration_num} saved: CSV at {csv_path}, PNG at {plot_path}"
        )

    print("\n========== Finalizing pruning ==========")
    model = make_pruning_permanent(model)
    # make_pruning_permanent moves to CPU, ensure it's back on device for final validation
    model = model.to(device)
    model_cpu = model.cpu() # For MACs/Params count on CPU
    example_inputs_cpu = example_inputs.cpu()
    final_macs, final_params = tp.utils.count_ops_and_params(
        model_cpu, example_inputs_cpu
    )
    final_param_reduction = (
        (base_nparams - final_params) / base_nparams if base_nparams > 0 else 0.0
    )
    final_mac_reduction = (base_macs - final_macs) / base_macs if base_macs > 0 else 0.0
    print(
        f"After final prune: MACs={final_macs / 1e9:.4f}G "
        f"(-{final_mac_reduction * 100:.2f}%), Params={final_params / 1e6:.4f}M "
        f"(-{final_param_reduction * 100:.2f}%)"
    )

    print("Validating model after permanent pruning ...")
    model.to(device) # Ensure model is on device for validation
    val_acc_final, miou_final, val_loss_final = validate(
        model, val_loader, criterion, device, config.num_classes, epoch=-1
    )
    print(
        f"Final model: Pixel Acc={val_acc_final:.4f}, mIoU={miou_final:.4f}, Loss={val_loss_final:.4f}"
    )

    final_name = (
        f"pruned_final_paramRed_{final_param_reduction:.2f}_miou_{miou_final:.3f}.pth"
    )
    onnx_filename = (
        f"pruned_final_paramRed_{final_param_reduction:.2f}_miou_{miou_final:.3f}.onnx"
    )
    final_path = os.path.join(base_save_dir, final_name)
    onnx_path = os.path.join(base_save_dir, onnx_filename)

    torch.save(model.state_dict(), final_path)

    sample_inputs_onnx = example_inputs.cpu() # ONNX export often prefers CPU inputs

    torch.onnx.export(
        model.cpu(), # Export CPU version of the model
        sample_inputs_onnx,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )

    model_size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"Saved final pruned model at: {final_path} ({model_size_mb:.2f} MB)")

    if all_train_losses:
        summary_stem = f"pruning_summary_{original_basename}"
        summary_csv = os.path.join(base_save_dir, f"{summary_stem}.csv")
        summary_png = os.path.join(base_save_dir, f"{summary_stem}.png")
        save_metrics_plot(
            epochs_range=range(1, len(all_train_losses) + 1),
            train_losses=all_train_losses,
            val_losses=all_val_losses,
            val_accuracies=all_val_accuracies,
            val_mious=all_val_mious,
            plot_path=summary_png,
        )
        save_metrics_to_csv(
            metrics_path=summary_csv,
            time_list=all_times,
            train_losses=all_train_losses,
            val_losses=all_val_losses,
            val_accuracies=all_val_accuracies,
            val_mious=all_val_mious,
            model_size=model_size_mb,
        )
        print(
            f"Saved overall pruning metrics: CSV at {summary_csv}, PNG at {summary_png}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pruning script for segmentation models."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the .pth model file to prune."
    )
    parser.add_argument("--datapath", help="Override path to dataset root directory.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for fine-tuning after each prune step.",
    )
    parser.add_argument(
        "--iterative_steps",
        type=int,
        default=1,
        help="Number of iterative pruning steps.",
    )
    parser.add_argument(
        "--target_prune_rate",
        type=float,
        default=0.4,
        help="Total filter reduction ratio (0.0-1.0), e.g., 0.4 for 40% reduction.",
    )
    parser.add_argument(
        "--max_map_drop",
        type=float,
        default=0.2,
        help="Maximum allowed mIoU drop relative to the original.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prune_main(args)


if __name__ == "__main__":
    main()
