import torch
import wandb
from torch import nn

from bonfire.model import aggregator as agg
from bonfire.model import modules as mod
from bonfire.model.models import MultipleInstanceNN
from dgr_luc_dataset import DgrLucDataset
from matplotlib import pyplot as plt

from torchvision.transforms import Resize


def get_model_param(key):
    return wandb.config[key]


DGR_D_CONV_OUT = 2744
DGR_D_ENC = 128
DGR_DS_ENC_HID = (512,)
DGR_DS_AGG_HID = (64,)


class DgrEncoderMultiResEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        conv3 = mod.ConvBlock(c_in=48, c_out=56, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2, conv3)
        self.fc_stack = mod.FullyConnectedStack(DGR_D_CONV_OUT, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrMultiResNN(MultipleInstanceNN):

    name = "MultiResNN"

    def __init__(self, device):
        super().__init__(device, DgrLucDataset.n_classes, DgrLucDataset.n_expected_dims)
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")

        self.s0_encoder = DgrEncoderMultiResEncoder(dropout)
        self.s1_encoder = DgrEncoderMultiResEncoder(dropout)
        self.s2_encoder = DgrEncoderMultiResEncoder(dropout)

        self.s0_aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrLucDataset.n_classes,
                                                    dropout, agg_func)
        self.s1_aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrLucDataset.n_classes,
                                                    dropout, agg_func)
        self.s2_aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrLucDataset.n_classes,
                                                    dropout, agg_func)
        self.sm_aggregator = agg.InstanceAggregator(DGR_D_ENC * 3, DGR_DS_AGG_HID, DgrLucDataset.n_classes,
                                                    dropout, agg_func)

        self.patch_transform = Resize(76)

    def _apply_patch_transform(self, instance_grid):
        # Instance grid should be 5d: P_X x P_Y x C x H x W
        g, c, h, w = instance_grid.shape[0], instance_grid.shape[2], instance_grid.shape[3], instance_grid.shape[4]
        # Flatten first
        flat_grid = torch.reshape(instance_grid, (-1, c, h, w))
        # Apply transformation (resizing)
        transformed_patches = self.patch_transform(flat_grid)
        # Arrange back to grid
        grid_out = torch.reshape(transformed_patches, (g, g, c, 76, 76))
        return grid_out

    @staticmethod
    def _reconstruct_img_from_grid(grid_patches, grid_size, patch_size=76):
        reconstruction = torch.zeros(patch_size * grid_size, patch_size * grid_size, 3)
        for x in range(grid_size):
            for y in range(grid_size):
                reconstruction[patch_size * x:patch_size * (x + 1), patch_size * y:patch_size * (y + 1), :] = \
                    torch.permute(grid_patches[x, y], (1, 2, 0))
        return reconstruction

    def _show_reconstructions(self, orig_instances, s0_instances, s1_instances, s2_instances):
        orig_reconstruction = self._reconstruct_img_from_grid(orig_instances, 8, patch_size=304)
        s0_reconstruction = self._reconstruct_img_from_grid(s0_instances, 8)
        s1_reconstruction = self._reconstruct_img_from_grid(s1_instances, 16)
        s2_reconstruction = self._reconstruct_img_from_grid(s2_instances, 32)

        fig, axes = plt.subplots(nrows=1, ncols=4)
        axes[0].imshow(orig_reconstruction.detach().cpu())
        axes[1].imshow(s0_reconstruction.detach().cpu())
        axes[2].imshow(s1_reconstruction.detach().cpu())
        axes[3].imshow(s2_reconstruction.detach().cpu())
        plt.show()

    def _internal_forward(self, bags):
        batch_size = len(bags)
        # We make four predictions for each bag: s0, s1, s2, main
        bag_predictions = torch.zeros((batch_size, 4, self.n_classes)).to(self.device)
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        all_instance_predictions = []
        for i, instances in enumerate(bags):
            # Move instances to gpu (if available)
            instances = instances.to(self.device)
            n_instances, n_channels = instances.shape[0], instances.shape[1]

            # Reshape instances to grid
            instances = instances.reshape(8, 8, n_channels, 304, 304)

            # Extract S0 instances - Cell size 304 x 304 px; 8 x 8 grid
            #  Resize to 76 x 76; Eff res 608 x 608 px (6.2%)
            s0_instances = self._apply_patch_transform(instances)

            # Extract S1 instances - Cell size 152 x 152 px; 16 x 16 grid
            #  Resize to 76 x 76; Eff res 1,216 * 1,216 px (24.7%)
            s1_splits = torch.zeros(16, 16, n_channels, 152, 152).to(self.device)
            s1_splits[::2, ::2, :, :, :] = instances[:, :, :, :152, :152]
            s1_splits[::2, 1::2, :, :, :] = instances[:, :, :, :152, 152:]
            s1_splits[1::2, ::2, :, :, :] = instances[:, :, :, 152:, :152]
            s1_splits[1::2, 1::2, :, :, :] = instances[:, :, :, 152:, 152:]
            s1_instances = self._apply_patch_transform(s1_splits)

            # Extract S2 instances - Cell size 76 x 76 px; 32 x 32 grid
            #  No resizing needed; Eff res 2432 x 2432 px (98.7%)
            s2_instances = torch.zeros(32, 32, n_channels, 76, 76).to(self.device)
            for x in range(4):
                for y in range(4):
                    s2_instances[x::4, y::4, :, :, :] = \
                        instances[:, :, :, 76 * x:76 * (x + 1), 76 * y:76 * (y + 1)]

            # Encode s0, s1, and s2 patches
            s0_embeddings = self.s0_encoder(torch.reshape(s0_instances, (-1, 3, 76, 76)))
            s1_embeddings = self.s1_encoder(torch.reshape(s1_instances, (-1, 3, 76, 76)))
            s2_embeddings = self.s2_encoder(torch.reshape(s2_instances, (-1, 3, 76, 76)))

            # Create main embeddings (sm) by stacking s0, s1, and s2 embedding
            s0_embeddings_r = torch.repeat_interleave(s0_embeddings, 16, dim=0)
            s1_embeddings_r = torch.repeat_interleave(s1_embeddings, 4, dim=0)
            sm_embeddings = torch.cat([s0_embeddings_r, s1_embeddings_r, s2_embeddings], dim=1)

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

    @staticmethod
    def _reshape_instance_preds(ins_preds):
        # Class first
        ins_preds = ins_preds.swapaxes(0, 1)
        # Reshape to grid; assumes square grid
        grid_size = int(ins_preds.shape[1] ** 0.5)
        return ins_preds.reshape(-1, grid_size, grid_size)
