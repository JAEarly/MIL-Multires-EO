"""
Adapted from https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from bonfire.model import models
from dgr_luc_dataset import DgrLucDataset


def get_model_param(key):
    return wandb.config[key]


class DgrUNet(models.MultipleInstanceNN):

    name = "DgrUNet"

    def __init__(self, device):
        bilinear = get_model_param("bilinear")
        out_func_name = get_model_param("out_func")
        super().__init__(device, DgrLucDataset.n_classes, DgrLucDataset.n_expected_dims)

        # Model
        factor = 2 if bilinear else 1
        self.in_conv = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Classifier
        if out_func_name == 'avg':
            self.out = OutConvAvg(64, self.n_classes)
        elif out_func_name == 'gap':
            self.out = OutGAP(64, self.n_classes)
        else:
            raise ValueError('Invalid out function: {:}'.format(out_func_name))

    def _internal_forward(self, bags):
        batch_size = len(bags)
        bag_predictions = torch.zeros((batch_size, self.n_classes)).to(self.device)
        bag_instance_predictions = []
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        for i, instances in enumerate(bags):
            # Should only be a single instance for this type of model
            assert len(instances) == 1
            x = instances[0]

            # Pass through model
            x = x.to(self.device).unsqueeze(0)
            # print('in', x.shape)
            x1 = self.in_conv(x)
            # print('x1', x1.shape)
            x2 = self.down1(x1)
            # print('x2', x2.shape)
            x3 = self.down2(x2)
            # print('x3', x3.shape)
            x4 = self.down3(x3)
            # print('x4', x4.shape)
            x5 = self.down4(x4)
            # print('x5', x5.shape)
            x = self.up1(x5, x4)
            # print('x up 1', x.shape)
            x = self.up2(x, x3)
            # print('x up 2', x.shape)
            x = self.up3(x, x2)
            # print('x up 3', x.shape)
            x = self.up4(x, x1)
            # print('x up 4', x.shape)
            bag_pred = self.out(x)

            # Update outputs
            bag_predictions[i] = bag_pred
            # TODO save segmentation outputs
            bag_instance_predictions.append(None)
        return bag_predictions, bag_instance_predictions


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff_x = x2.size()[3] - x1.size()[3]
        diff_y = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvAvg(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        clz_x = self.conv(x)
        # Calculate mean over all pixels to get class predictions
        bag_pred = torch.mean(clz_x, dim=(2, 3))
        return bag_pred


class OutGAP(nn.Module):

    def __init__(self, fc_in, n_classes):
        super().__init__()
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x):
        gap_x = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        bag_pred = self.fc(gap_x)
        return bag_pred
