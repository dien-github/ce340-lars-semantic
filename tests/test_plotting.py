import os
from train_code.utils.plotting import save_metrics_plot, save_metrics_to_csv

def test_save_metrics_plot_and_csv(tmp_path):
    epochs = 3
    train_losses = [1.0, 0.8, 0.6]
    val_losses = [1.1, 0.9, 0.7]
    val_accuracies = [0.5, 0.6, 0.7]
    val_mious = [0.3, 0.4, 0.5]
    plot_path = tmp_path / "metrics.png"
    csv_path = tmp_path / "metrics.csv"
    save_metrics_plot(range(1, epochs+1), train_losses, val_losses, val_accuracies, val_mious, str(plot_path))
    save_metrics_to_csv(str(csv_path), train_losses, val_losses, val_accuracies, val_mious)
    assert os.path.exists(plot_path)
    assert os.path.exists(csv_path)