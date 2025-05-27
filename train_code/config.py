import glob
import os
import torch
from datetime import datetime

class Config:
    seed = 2025
    num_classes = 3
    batch_size = 32
    epochs = 20
    learning_rate = 1e-4
    input_size = (320, 320)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark = True # Set to True for speed if input sizes are fixed

    dataset_path = "/home/grace/Documents/ce340-lars-semantic/LaRS_dataset"
    date_str = datetime.now().strftime("%Y%m%d")

    load_checkpoint_path = None # "/home/grace/Documents/ce340-lars-semantic/checkpoints/best_model.pth"

    @property
    def run_id(self):
        checkpoint_dir = f"checkpoints/train/{self.date_str}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        pattern = os.path.join(checkpoint_dir, "best_model_*.pth")
        files = glob.glob(pattern)
        ids = []
        for f in files:
            # best_model_20240528_1.pth
            base = os.path.basename(f)
            parts = base.replace(".pth", "").split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                ids.append(int(parts[-1]))
        return max(ids, default=0) + 1

    @property
    def base_model_name(self):
        return f"best_model_{self.date_str}_{self.run_id}"

    @property
    def best_model_path(self):
        return f"checkpoints/train/{self.date_str}/{self.base_model_name}.pth"
    
    def pruned_model_path(self, base_model_name=None):
        if base_model_name is None:
            base_model_name = self.base_model_name
        return f"checkpoints/train/{self.date_str}/{base_model_name}_pruned.pth"

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