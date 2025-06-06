import glob
import os
import platform
import torch
from datetime import datetime


class Config:
    seed = 2025
    num_classes = 3
    batch_size = 32
    num_workers = 4
    epochs = 20
    learning_rate = 1e-4
    patience = 7
    input_size = (320, 320)
    model_type = "lraspp"  # "lraspp" or "deeplab"
    freeze_layers = ["backbone"]  # None, "backbone", "classifier", or "all"
    unfreeze_layers = ["classifier"]  # None, "backbone", "classifier", or "all"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type = "combined"  # cross_entropy, dice, or combined
    ce_weight = 1.0
    dice_weight = 1.0 
    lap_weight = 0.5
    cpu_info = platform.processor().lower()
    use_ipex = "intel" in cpu_info
    cudnn_benchmark = True  # Set to True for speed if input sizes are fixed

    # dataset_path will be set by the command line argument

    load_checkpoint_path = None

    def __init__(self, dataset_path_override=None):
        # Allow overriding dataset_path, primarily for Kaggle/Colab environments
        # If not overridden, it will try to use a default or expect it to be set by args later.
        # For your Kaggle case, main.py will set this.
        self.dataset_path = dataset_path_override if dataset_path_override else "/home/grace/Documents/ce340-lars-semantic/LaRS_dataset" # Default local path

        # Update paths to be relative to self.dataset_path
        self.val_dataset_path = os.path.join(self.dataset_path, "lars_v1.0.0_images", "val", "images")
        self.train_dataset_path = os.path.join(self.dataset_path, "lars_v1.0.0_images", "train", "images")

        self.val_mask_path = os.path.join(self.dataset_path, "lars_v1.0.0_annotations", "val", "semantic_masks")
        self.train_mask_path = os.path.join(self.dataset_path, "lars_v1.0.0_annotations", "train", "semantic_masks")

        self.train_image_list_file = os.path.join(self.dataset_path, "lars_v1.0.0_images", "train", "image_list.txt")
        self.val_image_list_file = os.path.join(self.dataset_path, "lars_v1.0.0_images", "val", "image_list.txt")

        self.date_str = datetime.now().strftime("%Y%m%d")
        self._run_id = None
        self._base_model_name = None
        self._train_names = None
        self._val_names = None

    @property
    def run_id(self):
        if self._run_id is None:
            checkpoint_dir = f"checkpoints/train/{self.date_str}"
            os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure dir exists before globbing
            pattern = os.path.join(checkpoint_dir, f"best_model_{self.date_str}_*.pth")
            files = glob.glob(pattern)
            ids = []
            for f_path in files:
                base = os.path.basename(f_path)
                # Example: best_model_20240528_1.pth
                parts = base.replace(".pth", "").split("_")
                # Expecting format: best_model_YYYYMMDD_ID.pth
                if len(parts) == 4 and parts[0] == "best" and parts[1] == "model" and parts[2] == self.date_str and parts[3].isdigit():
                    ids.append(int(parts[3]))
                elif len(parts) >= 3 and parts[-1].isdigit(): # Fallback for potentially older naming
                    print(f"Warning: Found model with non-standard name: {base}. Attempting to parse ID.")
                    ids.append(int(parts[-1]))
            self._run_id = max(ids, default=0) + 1
        return self._run_id

    @property
    def base_model_name(self):
        if self._base_model_name is None:
            # Access self.run_id to ensure it's calculated and cached
            current_run_id = self.run_id
            self._base_model_name = f"best_model_{self.date_str}_{current_run_id}"
        return self._base_model_name

    @property
    def best_model_path(self):
        return f"checkpoints/train/{self.date_str}/{self.base_model_name}.pth"

    def pruned_model_path(self, base_model_name=None, iteration=None):
        # This method provides a naming convention. Pruner.py may use its own more detailed naming.
        if base_model_name is None:
            base_model_name = self.base_model_name
        return f"checkpoints/prune/{base_model_name}/prune_{iteration}.pth"

    def quantized_model_path(self, base_model_name=None):
        if base_model_name is None:
            base_model_name = self.base_model_name
        return f"checkpoints/train/{self.date_str}/{base_model_name}_quantized.pth"

    def tflite_model_path(self, base_model_name=None):
        if base_model_name is None:
            base_model_name = self.base_model_name
        return f"checkpoints/train/{self.date_str}/{base_model_name}.tflite"

    def onnx_model_path(self, base_model_name=None):
        if base_model_name is None:
            base_model_name = self.base_model_name
        return f"checkpoints/train/{self.date_str}/{base_model_name}.onnx"

    @property
    def metrics_path(self):
        return f"output/train/{self.date_str}/{self.base_model_name}_metrics.csv"

    @property
    def plots_path(self):
        return f"output/train/{self.date_str}/{self.base_model_name}_metrics.png"

    @property
    def train_names(self):
        if self._train_names is None:
            if not os.path.exists(self.train_image_list_file):
                raise FileNotFoundError(f"Train image list file not found: {self.train_image_list_file}")
            with open(self.train_image_list_file, encoding="utf-8") as f:
                self._train_names = [line.strip() for line in f]
        return self._train_names

    @property
    def val_names(self):
        if self._val_names is None:
            if not os.path.exists(self.val_image_list_file):
                raise FileNotFoundError(f"Validation image list file not found: {self.val_image_list_file}")
            with open(self.val_image_list_file, encoding="utf-8") as f:
                self._val_names = [line.strip() for line in f]
        return self._val_names
