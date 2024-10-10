import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_v

class DoubleConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        return x

class MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        return self.max_pool(x)

class UpConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_conv(x)

class CopyAndConcatenate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x_cut = F_v.center_crop(x2, [x1.shape[2], x1.shape[3]])
        return torch.cat([x1, x_cut], dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.d_conv_1 = DoubleConvolutional(in_channels, 64)
        self.d_conv_2 = DoubleConvolutional(64, 128)
        self.d_conv_3 = DoubleConvolutional(128, 256)
        self.d_conv_4 = DoubleConvolutional(256, 512)
        self.d_conv_5 = DoubleConvolutional(512, 1024)

        self.max_pool = MaxPool(2)

        self.up_conv_1 = UpConvolutional(1024, 512)
        self.up_conv_2 = UpConvolutional(512, 256)
        self.up_conv_3 = UpConvolutional(256, 128)
        self.up_conv_4 = UpConvolutional(128, 64)

        self.copy_and_concatenate = CopyAndConcatenate()

        self.u_conv_1 = DoubleConvolutional(1024, 512)
        self.u_conv_2 = DoubleConvolutional(512, 256)
        self.u_conv_3 = DoubleConvolutional(256, 128)
        self.u_conv_4 = DoubleConvolutional(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.d_conv_1(x)
        x = self.max_pool(x1)

        x2 = self.d_conv_2(x)
        x = self.max_pool(x2)

        x3 = self.d_conv_3(x)
        x = self.max_pool(x3)

        x4 = self.d_conv_4(x)
        x = self.max_pool(x4)

        x = self.d_conv_5(x)

        x = self.up_conv_1(x)
        x = self.copy_and_concatenate(x, x4)
        x = self.u_conv_1(x)

        x = self.up_conv_2(x)
        x = self.copy_and_concatenate(x, x3)
        x = self.u_conv_2(x)

        x = self.up_conv_3(x)
        x = self.copy_and_concatenate(x, x2)
        x = self.u_conv_3(x)

        x = self.up_conv_4(x)
        x = self.copy_and_concatenate(x, x1)
        x = self.u_conv_4(x)

        x = self.output(x)
        return x
