import torch

def pixel_accuracy_score(pred, target):
    """
    Computes pixel accuracy for multi-class segmentation.
    
    Parameters:
    - pred (torch.Tensor): The predicted segmentation map (batch_size, H, W).
    - target (torch.Tensor): The ground truth segmentation map (batch_size, H, W).

    Returns:
    - accuracy (float): The overall pixel accuracy.
    """
    correct = torch.eq(pred, target).int().sum().item()  # Correct predictions
    total = target.numel()  # Total number of pixels
    return correct / total

def iou_score(pred, target, num_classes=8):
    """
    Computes the mean IoU (Intersection over Union) for multi-class segmentation.
    
    Parameters:
    - pred (torch.Tensor): The predicted segmentation map (batch_size, H, W). Contains class indices.
    - target (torch.Tensor): The ground truth segmentation map (batch_size, H, W). Contains class indices.
    - num_classes (int): The number of segmentation classes.

    Returns:
    - mean_iou (float): The mean IoU across all classes.
    """
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_class = (pred == cls)
        target_class = (target == cls)
        
        intersection = torch.logical_and(pred_class, target_class).sum().item()
        union = torch.logical_or(pred_class, target_class).sum().item()
        
        if union == 0:
            iou = float('nan')  # If there is no union, skip this class
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    # Convert list to a tensor for further operations (e.g., averaging)
    iou_per_class = torch.tensor(iou_per_class)
    
    # Compute the mean IoU across all valid classes (ignore NaN values)
    mean_iou = torch.nanmean(iou_per_class).item()
    
    return mean_iou

