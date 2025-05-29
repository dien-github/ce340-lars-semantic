import matplotlib.pyplot as plt
import numpy as np
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

def save_metrics_to_csv(metrics_path, time_list, train_losses, val_losses, val_accuracies, val_mious, model_size=None):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True) # Ensure directory exists
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'mIoU'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                time_list[epoch],
                train_losses[epoch],
                val_losses[epoch],
                val_accuracies[epoch],
                val_mious[epoch]])
        if model_size is not None:
            writer.writerow(['Model Size (MB)', model_size])

def save_confusion_matrix(cm, class_names, output_path):
    """
    Save confusion matrix as an image file.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
        output_path (str): Path to save the confusion matrix image.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()