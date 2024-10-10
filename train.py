import argparse

import mlflow
import mlflow.pytorch

import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

from data import CityScapesDataset
from model_unet import UNet
from trainer import train, val

torch.manual_seed(1)

def main(args):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = CityScapesDataset(args.data_dir, kind='train', transform=transform)
    val_dataset = CityScapesDataset(args.data_dir, kind='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device.type.upper()}')

    model = torch.nn.DataParallel(UNet(in_channels=3, out_channels=args.num_classes), device_ids=[0, 1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        for epoch in range(args.epochs):
            training_loss = train(model, train_loader, optimiser, criterion, device)
            validation_loss, eval_metrics = val(model, val_loader, criterion, device)

            print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {training_loss:.4f} - Val Loss: {validation_loss:.4f} - Pixel Acc: {eval_metrics["pixel_accuracy_val"]:.4f} - Mean IoU: {eval_metrics["mean_iou_val"]:.4f}')

            mlflow.log_metric('training_loss', training_loss, step=epoch)
            mlflow.log_metric('validation_loss', validation_loss, step=epoch)
            mlflow.log_metrics(eval_metrics, step=epoch)

        scheduler.step()

        torch.save(model.module.state_dict(), f'{args.model_dir}/state_dict.pth')
        mlflow.pytorch.log_state_dict(model.module.state_dict(), artifact_path='model')
        mlflow.pytorch.log_model(model, artifact_path='model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--experiment_name', type=str, default='image-segmentation')
    args = parser.parse_args()

    main(args)
