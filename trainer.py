import torch

from evaluate import pixel_accuracy_score, iou_score

def train(model, train_loader, optimiser, criterion, device):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data['image'].to(device), data['mask'].to(device)

        optimiser.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def val(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    pac = 0.0
    miou = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['image'].to(device), data['mask'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            outputs_ = outputs.detach().argmax(dim=1)
            pac += pixel_accuracy_score(outputs_, labels)
            miou += iou_score(outputs_, labels)

    return running_loss / len(val_loader), {
        'pixel_accuracy_val': pac / len(val_loader),
        'mean_iou_val': miou / len(val_loader)
    }
