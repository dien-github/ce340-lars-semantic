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
        header = ['Epoch', 'Time (s)', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'mIoU']
        writer.writerow(header)

        for epoch in range(len(train_losses)):
            # Ensure all lists are of the same length as train_losses
            time_val = f"{time_list[epoch]:.2f}" if epoch < len(time_list) and isinstance(time_list[epoch], float) else (time_list[epoch] if epoch < len(time_list) else 'N/A')
            val_loss_val = val_losses[epoch] if epoch < len(val_losses) else 'N/A'
            val_acc_val = val_accuracies[epoch] if epoch < len(val_accuracies) else 'N/A'
            val_miou_val = val_mious[epoch] if epoch < len(val_mious) else 'N/A'
            
            writer.writerow([
                epoch + 1,
                time_val,
                train_losses[epoch],
                val_loss_val,
                val_acc_val,
                val_miou_val
            ])
        
        if model_size is not None:
            writer.writerow([]) # Add an empty line for separation
            writer.writerow(['Final Model Size (MB)', f"{model_size:.2f}"])

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