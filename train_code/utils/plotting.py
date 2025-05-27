import matplotlib.pyplot as plt
import csv
import os

def save_metrics_plot(epochs_range, train_losses, val_losses, val_accuracies, val_mious, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True) # Ensure directory exists
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Pixel Accuracy', marker='o', color='green')
    plt.title('Validation Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_mious, label='Validation mIoU', marker='o', color='red')
    plt.title('Validation Mean IoU (mIoU)')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory

def save_metrics_to_csv(metrics_path, train_losses, val_losses, val_accuracies, val_mious):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True) # Ensure directory exists
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'mIoU'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                train_losses[epoch],
                val_losses[epoch],
                val_accuracies[epoch],
                val_mious[epoch]])