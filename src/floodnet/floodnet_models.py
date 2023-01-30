from abc import ABC

import torch
import wandb
from overrides import overrides
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod
from floodnet.floodnet_dataset import FloodNetDataset


def get_model_param(key):
    return wandb.config[key]


def get_model_clz(model_type):
    if 'small' in model_type:
        return FloodNetInstanceSpaceNNSmall
    elif 'medium' in model_type:
        return FloodNetInstanceSpaceNNMedium
    elif 'large' in model_type:
        return FloodNetInstanceSpaceNNLarge
    elif 'resnet' in model_type:
        return FloodNetResNet18
    # elif 'unet' in model_type:
    #     return DgrUNet
    # elif model_type == 'multi_res_single_out':
    #     return DgrMultiResSingleOutNN
    # elif model_type == 'multi_res_multi_out':
    #     return DgrMultiResMultiOutNN
    raise ValueError('No model class found for model type {:s}'.format(model_type))


def get_n_params(model_type):
    model_clz = get_model_clz(model_type)
    model = model_clz('cpu')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


DGR_D_CONV_OUT_SMALL = 1200
DGR_D_CONV_OUT_MEDIUM = 6912
DGR_D_CONV_OUT_LARGE = 2016
DGR_DS_ENC_HID = (512,)
DGR_D_ENC = 128
DGR_DS_AGG_HID = (64,)


class FloodNetSceneToPatchNN(models.InstanceSpaceNN, ABC):

    @overrides
    def _internal_forward(self, bags, bags_metadata=None):
        bag_predictions, bag_instance_predictions = super()._internal_forward(bags)
        bag_patch_instance_predictions = []
        for bag_idx, ins_preds in enumerate(bag_instance_predictions):
            # Make clz first dim rather than last
            patch_preds = ins_preds.swapaxes(0, 1)

            # Reshape to grid
            # TODO Use of bags metadata ignores batching
            patch_preds = patch_preds.reshape(-1, bags_metadata['grid_size_x'], bags_metadata['grid_size_y'])

            # Add to overall list
            bag_patch_instance_predictions.append(patch_preds)
        return bag_predictions, bag_patch_instance_predictions


class FloodNetEncoderSmall(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DGR_D_CONV_OUT_SMALL, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FloodNetEncoderMedium(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DGR_D_CONV_OUT_MEDIUM, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FloodNetEncoderLarge(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        conv3 = mod.ConvBlock(c_in=48, c_out=56, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2, conv3)
        self.fc_stack = mod.FullyConnectedStack(DGR_D_CONV_OUT_LARGE, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FloodNetInstanceSpaceNNSmall(FloodNetSceneToPatchNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = FloodNetEncoderSmall(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, FloodNetDataset.n_classes, dropout, agg_func)
        super().__init__(device, FloodNetDataset.n_classes, FloodNetDataset.n_expected_dims, encoder, aggregator)


class FloodNetInstanceSpaceNNMedium(FloodNetSceneToPatchNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = FloodNetEncoderMedium(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, FloodNetDataset.n_classes, dropout, agg_func)
        super().__init__(device, FloodNetDataset.n_classes, FloodNetDataset.n_expected_dims, encoder, aggregator)


class FloodNetInstanceSpaceNNLarge(FloodNetSceneToPatchNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = FloodNetEncoderLarge(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, FloodNetDataset.n_classes, dropout, agg_func)
        super().__init__(device, FloodNetDataset.n_classes, FloodNetDataset.n_expected_dims, encoder, aggregator)


class FloodNetResNet18(models.MultipleInstanceNN):

    name = "FloodNetResNet18"

    def __init__(self, device):
        super().__init__(device, FloodNetDataset.n_classes, FloodNetDataset.n_expected_dims)
        self.device = device
        # Create pretrained resnet18 model but swap last layer to output the correct number of classes
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)

    def _internal_forward(self, bags, bags_metadata=None):
        batch_size = len(bags)
        bag_predictions = torch.zeros((batch_size, self.n_classes)).to(self.device)
        bag_instance_predictions = []
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        for i, instances in enumerate(bags):
            # Should only be a single instance for this type of model
            assert len(instances) == 1
            instance = instances[0]

            # Make prediction
            instance = instance.to(self.device).unsqueeze(0)
            bag_prediction = self.model(instance)
            bag_predictions[i] = bag_prediction
            bag_instance_predictions.append(None)
        return bag_predictions, bag_instance_predictions
