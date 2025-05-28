import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.amp as amp
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import subprocess
import sys

from config import Config
from data.dataset import LaRSDataset
from model.deeplab import get_deeplab_model, get_lraspp_model

# Constants for test data (adjust if your structure is different)
TEST_DATA_ROOT = "/home/grace/Documents/ce340-lars-semantic/lars"


def get_test_data_paths(test_data_root):
    base_img_path = os.path.join(test_data_root)  # , "lars_v1.0.0_images", "test")
    base_ann_path = os.path.join(test_data_root)  # , "lars_v1.0.0_annotations", "test")

    test_images_dir = os.path.join(base_img_path, "images")
    test_masks_dir = os.path.join(base_ann_path, "semantic_masks")
    test_image_list_file = os.path.join(base_img_path, "image_list.txt")

    if not os.path.exists(test_image_list_file):
        # Fallback if image_list.txt is not present, try to list files directly
        # This assumes image names in images_dir correspond to mask names in masks_dir
        print(
            f"Warning: {test_image_list_file} not found. Trying to infer image list from directory."
        )
        if os.path.exists(test_images_dir):
            image_files = [
                f.split(".")[0]
                for f in os.listdir(test_images_dir)
                if f.endswith((".jpg", ".png"))
            ]
            if not image_files:
                raise FileNotFoundError(
                    f"No images found in {test_images_dir} and image_list.txt is missing."
                )
            return test_images_dir, test_masks_dir, image_files
        else:
            raise FileNotFoundError(
                f"Test images directory not found: {test_images_dir}"
            )

    with open(test_image_list_file, "r", encoding="utf-8") as f:
        test_image_names = [line.strip() for line in f]

    return test_images_dir, test_masks_dir, test_image_names


def calculate_dice_coefficient(predicted, target, num_classes, smooth=1e-6):
    dice_scores = []
    for cls_idx in range(num_classes):
        pred_flat = (predicted == cls_idx).reshape(-1)
        target_flat = (target == cls_idx).reshape(-1)
        intersection = (pred_flat & target_flat).sum().item()
        denominator = pred_flat.sum().item() + target_flat.sum().item()
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_scores.append(dice)
    return np.array(dice_scores)


def get_color_palette(num_classes):
    """Returns a color palette for visualizing segmentation masks."""
    palette = [[0, 0, 0]]  # Background
    if num_classes > 1:
        palette.append([255, 0, 0])  # Class 1: Red
    if num_classes > 2:
        palette.append([0, 255, 0])  # Class 2: Green
    if num_classes > 3:
        palette.append([0, 0, 255])  # Class 3: Blue
    # Add more colors if needed
    while len(palette) < num_classes:
        palette.append(list(np.random.choice(range(256), size=3)))
    return torch.tensor(palette, dtype=torch.uint8)


def mask_to_rgb(mask, color_palette):
    """Converts a segmentation mask to an RGB image using a color palette."""
    rgb_mask = torch.zeros(
        (mask.size(0), mask.size(1), 3), dtype=torch.uint8, device=mask.device
    )
    for cls_idx, color in enumerate(color_palette):
        rgb_mask[mask == cls_idx] = color
    return rgb_mask.cpu().numpy()


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_iou_distribution(per_class_iou_all_images, class_names, output_path):
    plt.figure(figsize=(12, 6))
    # Filter out classes with no IoU scores to avoid errors in boxplot
    data_to_plot = [iou_list for iou_list in per_class_iou_all_images if iou_list]
    labels_to_plot = [
        class_names[i]
        for i, iou_list in enumerate(per_class_iou_all_images)
        if iou_list
    ]

    if not data_to_plot:
        print("No IoU data to plot for distribution.")
        return

    plt.boxplot(data_to_plot, tick_labels=labels_to_plot)  # Sửa ở đây
    plt.xlabel("Class")
    plt.ylabel("IoU Score")
    plt.title("IoU Distribution per Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"IoU distribution plot saved to {output_path}")


def main(args):
    cfg = Config()

    # --- Setup ---
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = not cfg.cudnn_benchmark

    device = torch.device(cfg.device)

    model_path_to_test = args.model_path
    if not os.path.exists(model_path_to_test):
        print(f"Error: Model path not found: {model_path_to_test}")
        return

    model_name_tested = os.path.splitext(os.path.basename(model_path_to_test))[0]
    output_base_dir = os.path.join("output", "test", cfg.date_str, model_name_tested)
    visual_output_dir = os.path.join(output_base_dir, "visual_examples")
    os.makedirs(visual_output_dir, exist_ok=True)

    # --- Load Test Data ---
    try:
        test_images_dir, test_masks_dir, test_image_names = get_test_data_paths(
            args.test_data_root
        )
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    test_dataset = LaRSDataset(
        image_dir=test_images_dir,
        image_names=test_image_names,
        mask_dir=test_masks_dir,
        transform=None,  # Normalization can be added here if model expects it
        target_size=cfg.input_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # Use a batch size suitable for inference
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Test dataset loaded: {len(test_dataset)} images.")

    # --- Load Model ---
    # model = get_deeplab_model(num_classes=cfg.num_classes, device=device)
    model = get_lraspp_model(num_classes=cfg.num_classes, device=device)
    # model.load_state_dict(torch.load(model_path_to_test, map_location=device))
    state_dict = torch.load(model_path_to_test, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        print("Attempting to compile the model with torch.compile()...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Proceeding without compilation.")

    print(f"Model '{model_path_to_test}' loaded successfully.")

    # --- Evaluation ---
    all_preds, all_targets = [], []
    per_class_iou_all_images = [[] for _ in range(cfg.num_classes)]
    per_class_dice_all_images = [[] for _ in range(cfg.num_classes)]

    total_inference_time = 0
    num_processed_images = 0
    worst_iou_images = []  # Store (iou, image_id, pred_mask, gt_mask)

    color_palette = get_color_palette(cfg.num_classes)

    # Warm-up for accurate latency measurement
    if device.type == "cuda":
        print("Warming up GPU...")
        for _ in range(5):
            dummy_input = torch.randn(
                args.batch_size, 3, *cfg.input_size, device=device
            )
            with (
                torch.no_grad(),
                amp.autocast(device.type, enabled=(device.type == "cuda")),
            ):
                _ = model(dummy_input)
        torch.cuda.synchronize()

    with torch.no_grad():
        loop = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Evaluating Test Set"
        )
        for i, (images, masks) in loop:
            images, masks = images.to(device), masks.to(device)

            start_time = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()

            with amp.autocast(device.type, enabled=(device.type == "cuda")):
                outputs = model(images)["out"]

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_inference_time += end_time - start_time
            num_processed_images += images.size(0)

            _, predicted = torch.max(outputs.data, 1)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            for j in range(predicted.size(0)):  # Iterate over images in batch
                pred_single = predicted[j]
                mask_single = masks[j]
                image_id = test_dataset.image_names[i * args.batch_size + j]

                current_image_ious = []
                for cls_idx in range(cfg.num_classes):
                    intersection = (
                        ((pred_single == cls_idx) & (mask_single == cls_idx))
                        .sum()
                        .item()
                    )
                    union = (
                        ((pred_single == cls_idx) | (mask_single == cls_idx))
                        .sum()
                        .item()
                    )
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    per_class_iou_all_images[cls_idx].append(iou)
                    current_image_ious.append(iou)

                # Stress test: track images with low mean IoU
                mean_img_iou = np.mean(current_image_ious)
                if len(worst_iou_images) < 10 or mean_img_iou < worst_iou_images[-1][0]:
                    # Save masks for visualization later
                    worst_iou_images.append(
                        (mean_img_iou, image_id, pred_single.cpu(), mask_single.cpu())
                    )
                    worst_iou_images.sort(key=lambda x: x[0])
                    if len(worst_iou_images) > 10:
                        worst_iou_images.pop()

                dice_scores_single = calculate_dice_coefficient(
                    pred_single, mask_single, cfg.num_classes
                )
                for cls_idx in range(cfg.num_classes):
                    per_class_dice_all_images[cls_idx].append(
                        dice_scores_single[cls_idx]
                    )

                # Save some visual examples
                if (
                    i < args.num_visual_examples // args.batch_size + 1
                    and j < args.num_visual_examples % args.batch_size
                ):
                    original_pil_image, _ = test_dataset._load_image_mask_pil(
                        image_id
                    )  # Load original for context

                    pred_rgb = mask_to_rgb(pred_single, color_palette)
                    mask_rgb = mask_to_rgb(mask_single, color_palette)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(original_pil_image)
                    axes[0].set_title(f"Original: {image_id}")
                    axes[0].axis("off")
                    axes[1].imshow(mask_rgb)
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")
                    axes[2].imshow(pred_rgb)
                    axes[2].set_title("Prediction")
                    axes[2].axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(visual_output_dir, f"{image_id}_comparison.png")
                    )
                    plt.close()

    all_preds_np = np.concatenate([p.reshape(-1) for p in all_preds])
    all_targets_np = np.concatenate([t.reshape(-1) for t in all_targets])

    # --- Calculate Metrics ---
    pixel_accuracy = (all_preds_np == all_targets_np).mean()

    mean_iou_per_class = [
        np.mean(ious) if ious else 0 for ious in per_class_iou_all_images
    ]
    mIoU = np.mean(
        [miou for miou in mean_iou_per_class if miou > 0]
    )  # Exclude classes not present if desired

    mean_dice_per_class = [
        np.mean(dices) if dices else 0 for dices in per_class_dice_all_images
    ]
    mean_dice_coefficient = np.mean(
        [mdice for mdice in mean_dice_per_class if mdice > 0]
    )

    class_names = [
        f"Class {i}" for i in range(cfg.num_classes)
    ]  # Or provide actual names

    # --- Error Analysis ---
    # Confusion Matrix
    cm = confusion_matrix(
        all_targets_np, all_preds_np, labels=list(range(cfg.num_classes))
    )
    plot_confusion_matrix(
        cm, class_names, os.path.join(output_base_dir, "confusion_matrix.png")
    )

    # IoU Distribution
    plot_iou_distribution(
        per_class_iou_all_images,
        class_names,
        os.path.join(output_base_dir, "iou_distribution.png"),
    )

    # Visual inspection of worst cases (stress test)
    print("\nWorst performing images (by mean IoU):")
    for iou_val, img_id, pred_m, gt_m in worst_iou_images:
        print(f"  Image ID: {img_id}, Mean IoU: {iou_val:.4f}")
        # Save these specific worst cases
        original_pil_image, _ = test_dataset._load_image_mask_pil(img_id)
        pred_rgb = mask_to_rgb(pred_m, color_palette)
        gt_rgb = mask_to_rgb(gt_m, color_palette)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_pil_image)
        axes[0].set_title(f"Original: {img_id}")
        axes[0].axis("off")
        axes[1].imshow(gt_rgb)
        axes[1].set_title(f"Ground Truth (IoU: {iou_val:.3f})")
        axes[1].axis("off")
        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(visual_output_dir, f"worst_{img_id}_iou_{iou_val:.3f}.png")
        )
        plt.close()

    # --- Performance & Requirements ---
    model_size_mb = os.path.getsize(model_path_to_test) / (1024 * 1024)
    avg_latency_ms = (
        (total_inference_time / num_processed_images) * 1000
        if num_processed_images > 0
        else 0
    )
    fps = num_processed_images / total_inference_time if total_inference_time > 0 else 0

    # Requirement checks
    req_miou_pass = mIoU >= args.req_miou
    req_size_pass = model_size_mb <= args.req_model_size
    req_latency_pass = avg_latency_ms <= args.req_latency

    print("\n--- Test Results ---")
    print(f"Model: {model_path_to_test}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice_coefficient:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"  IoU for {class_name}: {mean_iou_per_class[i]:.4f}")
        print(f"  Dice for {class_name}: {mean_dice_per_class[i]:.4f}")

    print("\n--- Performance ---")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Average Latency: {avg_latency_ms:.2f} ms/image")
    print(f"Inference Speed: {fps:.2f} FPS")

    print("\n--- Requirement Checks ---")
    print(
        f"mIoU >= {args.req_miou}: {'PASS' if req_miou_pass else 'FAIL'} (Actual: {mIoU:.4f})"
    )
    print(
        f"Model Size <= {args.req_model_size} MB: {'PASS' if req_size_pass else 'FAIL'} (Actual: {model_size_mb:.2f} MB)"
    )
    print(
        f"Latency <= {args.req_latency} ms: {'PASS' if req_latency_pass else 'FAIL'} (Actual: {avg_latency_ms:.2f} ms)"
    )

    # --- Logging to CSV ---
    csv_path = os.path.join(output_base_dir, "test_summary.csv")
    csv_header = [
        "ModelName",
        "TestDate",
        "PixelAccuracy",
        "mIoU",
        "MeanDice",
        "ModelSizeMB",
        "AvgLatencyMs",
        "FPS",
        "Req_mIoU_Pass",
        "Req_Size_Pass",
        "Req_Latency_Pass",
    ]
    for i in range(cfg.num_classes):
        csv_header.extend([f"IoU_Class{i}", f"Dice_Class{i}"])

    csv_row = [
        model_name_tested,
        cfg.date_str,
        pixel_accuracy,
        mIoU,
        mean_dice_coefficient,
        model_size_mb,
        avg_latency_ms,
        fps,
        req_miou_pass,
        req_size_pass,
        req_latency_pass,
    ]
    for i in range(cfg.num_classes):
        csv_row.extend([mean_iou_per_class[i], mean_dice_per_class[i]])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerow(csv_row)
    print(f"\nTest summary saved to: {csv_path}")

    # Auto-run sum.py after test
    sum_script = "/home/grace/Documents/ce340-lars-semantic/sum.py"
    try:
        subprocess.run([sys.executable, sum_script], check=True)
        print("Auto-summarized all test_summary.csv files.")
    except Exception as e:
        print(f"Failed to run sum.py: {e}")

    print(f"\nVisualizations and detailed logs saved in: {output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a semantic segmentation model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth) file.",
    )
    parser.add_argument(
        "--test_data_root",
        type=str,
        default=TEST_DATA_ROOT,
        help="Root directory of the LaRS test dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for testing."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--num_visual_examples",
        type=int,
        default=5,
        help="Number of visual examples to save.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Enable torch.compile for the model (PyTorch 2.0+).",
    )

    # Requirement arguments
    parser.add_argument("--req_miou", type=float, default=0.9, help="Required mIoU.")
    parser.add_argument(
        "--req_model_size", type=float, default=5.0, help="Required model size in MB."
    )
    parser.add_argument(
        "--req_latency", type=float, default=200.0, help="Required latency in ms."
    )

    args = parser.parse_args()
    main(args)
