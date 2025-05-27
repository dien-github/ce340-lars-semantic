import torch

class Config:
    seed = 2025
    num_classes = 3
    batch_size = 64
    epochs = 10
    learning_rate = 1e-4
    input_size = (320, 320)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark = True # Set to True for speed if input sizes are fixed

    dataset_path = "LaRS_dataset"
    best_model_path = "ce340-lars-semantic/checkpoints/best_model.pth"
    metrics_path = "output/training_log.csv"
    plots_path = "output/training_metrics.png"