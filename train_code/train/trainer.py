from tqdm import tqdm
import numpy as np
import torch
import torch.amp as amp

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Train] Epoch {epoch}")
    
    for _, (images, masks) in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with amp.autocast('cuda', enabled=(device.type == 'cuda')): # Automatic Mixed Precision
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
        
        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        # For PyTorch >= 1.6, calls optimizer.zero_grad() internally if set_to_none=True
        scaler.scale(loss).backward()
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
        # Otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        if scheduler:
            scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_time = loop.format_dict['elapsed']
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")
    return epoch_time, epoch_loss

def validate(model, dataloader, criterion, device, num_classes, epoch):
    model.eval()
    correct_pixels, total_pixels, running_val_loss = 0, 0, 0.0
    iou_scores = [[] for _ in range(num_classes)]
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader), desc=f"[Val] Epoch {epoch}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            with amp.autocast('cuda', enabled=(device.type == 'cuda')): # AMP for validation
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
            
            _, predicted = torch.max(outputs.data, 1) # Use .data to avoid tracking history
            running_val_loss += loss.item() * images.size(0) # Accumulate loss scaled by batch size
            
            total_pixels += masks.numel()
            correct_pixels += (predicted == masks).sum().item()
            
            for cls in range(num_classes):
                intersection = ((predicted == cls) & (masks == cls)).sum().item()
                union = ((predicted == cls) | (masks == cls)).sum().item()
                if union > 0:
                    iou_scores[cls].append(intersection / union)
            
            loop.set_postfix(iou_scores=[np.mean(iou) if iou else 0 for iou in iou_scores])
    mean_iou = np.mean([np.mean(iou) if iou else 0 for iou in iou_scores])
    pixel_accuracy = correct_pixels / total_pixels
    epoch_val_loss = running_val_loss / len(dataloader.dataset)
    print(f"Validation accuracy: {pixel_accuracy:.4f}, Loss: {epoch_val_loss:.4f}, mIoU: {mean_iou:.4f}")
    return pixel_accuracy, mean_iou, epoch_val_loss