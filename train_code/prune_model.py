import argparse
import os
import torch
import torch.nn.utils.prune as prune

from config import Config
from model.deeplab import (
    get_deeplab_model,
)  # Giả sử get_deeplab_model có thể tải mô hình mà không cần device ban đầu


def prune_model_unstructured(model, layers_to_prune, amount=0.3):
    """
    Applies unstructured L1 magnitude pruning to specified layers of the model.

    Args:
        model (torch.nn.Module): The model to prune.
        layers_to_prune (list of tuple): List of (module, name) tuples to prune.
                                         Example: [(model.backbone.conv1, 'weight')]
        amount (float): The fraction of connections to prune (0.0 to 1.0).
    """
    for module, name in layers_to_prune:
        prune.l1_unstructured(module, name=name, amount=amount)
        # Make pruning permanent by removing the re-parameterization
        prune.remove(module, name)
    print(f"Pruned {len(layers_to_prune)} layers with amount {amount:.2f}")


def main(args):
    cfg = Config()
    device = torch.device(cfg.device)

    # Load the trained model
    if not args.model_path or not os.path.exists(args.model_path):
        print(
            f"Error: Original model path '{args.model_path}' not found or not specified."
        )
        print(
            "Please provide a valid path to a trained .pth model file using --model_path."
        )
        return

    print(f"Loading model from: {args.model_path}")
    # Tải mô hình với cấu trúc gốc trước, sau đó tải state_dict
    model = get_deeplab_model(num_classes=cfg.num_classes, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to eval mode for pruning
    print("Model loaded successfully.")

    # Identify layers to prune (e.g., Conv2d layers in the backbone and classifier)
    # Đây là một ví dụ, bạn có thể cần điều chỉnh dựa trên cấu trúc mô hình cụ thể của mình
    layers_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Chỉ tỉa các lớp Conv2d có trọng số (không phải bias)
            # và không phải là lớp cuối cùng của classifier nếu bạn muốn giữ nguyên nó
            # Ví dụ: không tỉa model.classifier[4] nếu đó là lớp output cuối cùng
            if 'classifier.4' not in name: # Ví dụ điều kiện không tỉa lớp cuối
                layers_to_prune.append((module, "weight"))

    if not layers_to_prune:
        print("No Conv2d layers found to prune. Exiting.")
        return

    print(f"Found {len(layers_to_prune)} Conv2d layers to potentially prune.")

    # Apply pruning
    prune_model_unstructured(model, layers_to_prune, amount=args.pruning_amount)

    # Save the pruned model
    # Sử dụng tên file gốc để tạo tên file pruned
    original_model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    pruned_model_save_path = cfg.pruned_model_path(
        base_model_name=original_model_basename
    )
    os.makedirs(os.path.dirname(pruned_model_save_path), exist_ok=True)
    torch.save(model.state_dict(), pruned_model_save_path)
    print(f"Pruned model saved to: {pruned_model_save_path}")
    print(
        f"Original model size: {os.path.getsize(args.model_path) / (1024 * 1024):.2f} MB"
    )
    print(
        f"Pruned model size: {os.path.getsize(pruned_model_save_path) / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune a trained DeepLabV3 model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained .pth model file to prune.",
    )
    parser.add_argument(
        "--pruning_amount",
        type=float,
        default=0.3,
        help="Fraction of weights to prune (0.0 to 1.0). Default is 0.3 (30%).",
    )
    args = parser.parse_args()
    main(args)
