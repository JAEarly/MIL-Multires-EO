import torch
import wandb
from torch import nn

from bonfire.model import aggregator as agg
from bonfire.model import modules as mod
from bonfire.model.models import MultipleInstanceNN
from floodnet.floodnet_dataset import FloodNetDataset
from matplotlib import pyplot as plt
from abc import ABC

from torchvision.transforms import Resize


def get_model_param(key):
    return wandb.config[key]


FLOODNET_D_CONV_OUT = 5600
FLOODNET_DS_ENC_HID = (512,)
FLOODNET_D_ENC = 128
FLOODNET_DS_AGG_HID = (64,)


class FloodNetEncoderMultiRes(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        conv3 = mod.ConvBlock(c_in=48, c_out=56, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2, conv3)
        self.fc_stack = mod.FullyConnectedStack(FLOODNET_D_CONV_OUT, FLOODNET_DS_ENC_HID, FLOODNET_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FloodNetMultiResNN(MultipleInstanceNN, ABC):

    def __init__(self, device):
        super().__init__(device, FloodNetDataset.n_classes, FloodNetDataset.n_expected_dims)
        self.dropout = get_model_param("dropout")
        self.agg_func = get_model_param("agg_func")

        self.s0_encoder = FloodNetEncoderMultiRes(self.dropout)
        self.s1_encoder = FloodNetEncoderMultiRes(self.dropout)
        self.s2_encoder = FloodNetEncoderMultiRes(self.dropout)

        self.sm_aggregator = agg.InstanceAggregator(FLOODNET_D_ENC * 3, FLOODNET_DS_AGG_HID, FloodNetDataset.n_classes,
                                                    self.dropout, self.agg_func)

        self.patch_transform = Resize(102)

    def _apply_patch_transform(self, instance_grid):
        # Instance grid should be 5d: P_X x P_Y x C x H x W
        p_x, p_y, c, h, w = instance_grid.shape
        # Flatten (removing p_x, p_y -> p_x * p_y)
        flat_grid = torch.reshape(instance_grid, (-1, c, h, w))
        # Apply transformation (resizing)
        transformed_patches = self.patch_transform(flat_grid)
        # Arrange back to grid
        grid_out = torch.reshape(transformed_patches, (p_x, p_y, c, 102, 102))
        return grid_out

    @staticmethod
    def _reconstruct_img_from_grid(grid_patches, grid_size_x, grid_size_y, patch_size=102):
        reconstruction = torch.zeros(patch_size * grid_size_x, patch_size * grid_size_y, 3)
        for x in range(grid_size_x):
            for y in range(grid_size_y):
                reconstruction[patch_size * x:patch_size * (x + 1), patch_size * y:patch_size * (y + 1), :] = \
                    torch.permute(grid_patches[x, y], (1, 2, 0))
        return reconstruction

    def _show_reconstructions(self, orig_img, orig_instances, s0_instances, s1_instances, s2_instances):
        orig_reconstruction = self._reconstruct_img_from_grid(orig_instances, 8, 6, patch_size=500)
        s0_reconstruction = self._reconstruct_img_from_grid(s0_instances, 8, 6)
        s1_reconstruction = self._reconstruct_img_from_grid(s1_instances, 16, 12)
        s2_reconstruction = self._reconstruct_img_from_grid(s2_instances, 32, 24)

        print(orig_img.width, orig_img.height)
        print(orig_reconstruction.shape)
        print(s0_reconstruction.shape)
        print(s1_reconstruction.shape)
        print(s2_reconstruction.shape)

        fig, axes = plt.subplots(nrows=1, ncols=5)
        axes[0].imshow(orig_img)
        axes[1].imshow(torch.transpose(orig_reconstruction.detach().cpu(), 0, 1))
        axes[2].imshow(torch.transpose(s0_reconstruction.detach().cpu(), 0, 1))
        axes[3].imshow(torch.transpose(s1_reconstruction.detach().cpu(), 0, 1))
        axes[4].imshow(torch.transpose(s2_reconstruction.detach().cpu(), 0, 1))
        plt.show()

    def _forward_encode(self, instances, bags_metadata):
        instances = instances.to(self.device)
        n_instances, n_channels = instances.shape[0], instances.shape[1]

        print(bags_metadata)

        from PIL import Image
        img = Image.open('data/FloodNet/train/train-org-img/{:d}.jpg'.format(bags_metadata['id'].item()))

        # TODO remove magic numbers
        # Reshape instances to grid
        instances = instances.reshape(8, 6, n_channels, 500, 500)

        # Extract S0 instances - Cell size 500 x 500 px; 8 x 6 grid
        #  Resize to 102 x 102 px; Eff res 816 x 612 px (4.2%)
        s0_instances = self._apply_patch_transform(instances)

        # Extract S1 instances - Cell size 250 x 250 px; 16 x 12 grid
        #  Resize to 102 x 102 px; Eff res 1,632 x 1,224 px (16.6%)
        s1_splits = torch.zeros(16, 12, n_channels, 250, 250).to(self.device)
        s1_splits[::2, ::2, :, :, :] = instances[:, :, :, :250, :250]
        s1_splits[::2, 1::2, :, :, :] = instances[:, :, :, :250, 250:]
        s1_splits[1::2, ::2, :, :, :] = instances[:, :, :, 250:, :250]
        s1_splits[1::2, 1::2, :, :, :] = instances[:, :, :, 250:, 250:]
        s1_instances = self._apply_patch_transform(s1_splits)

        # Extract S2 instances - Cell size 125 x 125 px; 32 x 24 grid
        #  Resize to 102 x 102 px; Eff res 3264 x 2448 px (66.6%)
        s2_splits = torch.zeros(32, 24, n_channels, 125, 125).to(self.device)
        for x in range(4):
            for y in range(4):
                s2_splits[x::4, y::4, :, :, :] = \
                    instances[:, :, :, 125 * x:125 * (x + 1), 125 * y:125 * (y + 1)]
        s2_instances = self._apply_patch_transform(s2_splits)

        self._show_reconstructions(img, instances, s0_instances, s1_instances, s2_instances)
        # self._show_reconstructions(img, instances, None, None, None)
        exit(0)

        # Encode s0, s1, and s2 patches
        s0_embeddings = self.s0_encoder(torch.reshape(s0_instances, (-1, 3, 102, 102)))
        s1_embeddings = self.s1_encoder(torch.reshape(s1_instances, (-1, 3, 102, 102)))
        s2_embeddings = self.s2_encoder(torch.reshape(s2_instances, (-1, 3, 102, 102)))

        # Create main embeddings (sm) by stacking s0, s1, and s2 embedding
        s0_embeddings_r = torch.repeat_interleave(s0_embeddings, 16, dim=0)
        s1_embeddings_r = torch.repeat_interleave(s1_embeddings, 4, dim=0)
        sm_embeddings = torch.cat([s0_embeddings_r, s1_embeddings_r, s2_embeddings], dim=1)

        # Return embeddings at three different scales + one combined scale
        return s0_embeddings, s1_embeddings, s2_embeddings, sm_embeddings

    @staticmethod
    def _reshape_instance_preds(ins_preds):
        # Class first
        ins_preds = ins_preds.swapaxes(0, 1)
        # Reshape to grid; assumes square grid
        grid_size = int(ins_preds.shape[1] ** 0.5)
        return ins_preds.reshape(-1, grid_size, grid_size)


class FloodNetMultiResSingleOutNN(FloodNetMultiResNN):

    name = "MultiResSingleOutNN"

    def _internal_forward(self, bags, bags_metadata=None):
        batch_size = len(bags)
        # We make a single prediction for each bag using the combined scale
        bag_predictions = torch.zeros((batch_size, self.n_classes)).to(self.device)
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        all_instance_predictions = []
        for i, instances in enumerate(bags):
            # Get combined embeddings
            _, _, _, sm_embeddings = self._forward_encode(instances, bags_metadata)

            # Classify instances and aggregate at each scale
            #  Here, the instance interpretations are actually predictions
            sm_bag_pred, sm_inst_preds = self.sm_aggregator(sm_embeddings)

            # Reshape instance preds to grid
            sm_inst_preds = self._reshape_instance_preds(sm_inst_preds)

            # Update bag outputs
            bag_predictions[i] = sm_bag_pred

            # Update instance outputs
            all_instance_predictions.append(sm_inst_preds)

        return bag_predictions, all_instance_predictions


class FloodNetMultiResMultiOutNN(FloodNetMultiResNN):

    name = "MultiResMultiOutNN"

    def __init__(self, device):
        super().__init__(device)
        self.s0_aggregator = agg.InstanceAggregator(FLOODNET_D_ENC, FLOODNET_DS_AGG_HID, FloodNetDataset.n_classes,
                                                    self.dropout, self.agg_func)
        self.s1_aggregator = agg.InstanceAggregator(FLOODNET_D_ENC, FLOODNET_DS_AGG_HID, FloodNetDataset.n_classes,
                                                    self.dropout, self.agg_func)
        self.s2_aggregator = agg.InstanceAggregator(FLOODNET_D_ENC, FLOODNET_DS_AGG_HID, FloodNetDataset.n_classes,
                                                    self.dropout, self.agg_func)

    def _internal_forward(self, bags, bags_metadata=None):
        batch_size = len(bags)
        # We make four predictions for each bag: s0, s1, s2, main
        bag_predictions = torch.zeros((batch_size, 4, self.n_classes)).to(self.device)
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        all_instance_predictions = []
        for i, instances in enumerate(bags):
            # Get embeddings at each scale
            s0_embeddings, s1_embeddings, s2_embeddings, sm_embeddings = self._forward_encode(instances)

            # Classify instances and aggregate at each scale
            #  Here, the instance interpretations are actually predictions
            s0_bag_pred, s0_inst_preds = self.s0_aggregator(s0_embeddings)
            s1_bag_pred, s1_inst_preds = self.s1_aggregator(s1_embeddings)
            s2_bag_pred, s2_inst_preds = self.s2_aggregator(s2_embeddings)
            sm_bag_pred, sm_inst_preds = self.sm_aggregator(sm_embeddings)

            # Reshape instance preds to grid
            s0_inst_preds = self._reshape_instance_preds(s0_inst_preds)
            s1_inst_preds = self._reshape_instance_preds(s1_inst_preds)
            s2_inst_preds = self._reshape_instance_preds(s2_inst_preds)
            sm_inst_preds = self._reshape_instance_preds(sm_inst_preds)

            # Update bag outputs
            bag_predictions[i, 0] = s0_bag_pred
            bag_predictions[i, 1] = s1_bag_pred
            bag_predictions[i, 2] = s2_bag_pred
            bag_predictions[i, 3] = sm_bag_pred

            # Update instance outputs
            bag_instance_predictions = [
                s0_inst_preds,
                s1_inst_preds,
                s2_inst_preds,
                sm_inst_preds,
            ]
            all_instance_predictions.append(bag_instance_predictions)

        return bag_predictions, all_instance_predictions
