import matplotlib.pylot as plt
import csv

def save_metrics_plot(epochs_range, train_losses, val_losses, val_accuracies, val_mious, plot_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Pixel Accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_mious, label='mIoU')
    plt.title('Mean IoU')
    plt.tight_layout()
    plt.savefig(plot_path)

def save_metrics_to_csv(metrics_path, train_losses, val_losses, val_accuracies, val_mious):
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'mIoU'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch], val_accuracies[epoch], val_mious[epoch]])