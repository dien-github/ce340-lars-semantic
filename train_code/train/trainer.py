from tqdm import tqdm
import numpy as np
import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Train] Epoch {epoch}")
    
    for _, (images, masks) in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device, num_classes, epoch):
    model.eval()
    correct, total, running_val_loss = 0, 0, 0.0
    iou_scores = [[] for _ in range(num_classes)]
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader), desc=f"[Val] Epoch {epoch}")
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, masks)
            running_val_loss += loss.item()
            
            total += masks.numel()
            correct += (predicted == masks).sum().item()
            
            for cls in range(num_classes):
                intersection = ((predicted == cls) & (masks == cls)).sum().item()
                union = ((predicted == cls) | (masks == cls)).sum().item()
                if union > 0:
                    iou_scores[cls].append(intersection / union)
            
            loop.set_postfix(iou_scores=[np.mean(iou) if iou else 0 for iou in iou_scores])
    mean_iou = np.mean([np.mean(iou) if iou else 0 for iou in iou_scores])
    accuracy = correct / total
    epoch_val_loss = running_val_loss / len(dataloader.dataset)
    return accuracy, mean_iou, epoch_val_loss