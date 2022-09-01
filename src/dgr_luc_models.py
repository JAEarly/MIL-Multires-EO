import torch
import wandb
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod
from dgr_luc_dataset import DgrLucDataset


def get_model_param(key):
    return wandb.config[key]


def _num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


DGR_D_ENC = 128
DGR_DS_ENC_HID = (512,)
DGR_DS_AGG_HID = (64,)


class DgrEncoderSmall(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DgrLucDataset.d_in, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrEncoderMedium(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        conv3 = mod.ConvBlock(c_in=48, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2, conv3)
        self.fc_stack = mod.FullyConnectedStack(DgrLucDataset.d_in, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrInstanceSpaceNNSmall(models.InstanceSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = DgrEncoderSmall(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrLucDataset.n_classes, dropout, agg_func)
        super().__init__(device, DgrLucDataset.n_classes, DgrLucDataset.n_expected_dims, encoder, aggregator)
        print('Num params: {:d}'.format(_num_params(self)))


class DgrInstanceSpaceNNMedium(models.InstanceSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = DgrEncoderMedium(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrLucDataset.n_classes, dropout, agg_func)
        super().__init__(device, DgrLucDataset.n_classes, DgrLucDataset.n_expected_dims, encoder, aggregator)
        print('Num params: {:d}'.format(_num_params(self)))


class DgrResNet18(models.MultipleInstanceNN):

    name = "DgrResNet18"

    def __init__(self, device):
        super().__init__(device, DgrLucDataset.n_classes, DgrLucDataset.n_expected_dims)
        self.device = device
        # Create pretrained resnet18 model but swap last layer to output the correct number of classes
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
        print('Num params: {:d}'.format(_num_params(self.model)))

    def _internal_forward(self, bags):
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
            bag_instance_predictions.append([None])
        return bag_predictions, bag_instance_predictions
