import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from dgr_luc_dataset import RECONSTRUCTION_DATA_DIR_FMT


class MilLucInterpretabilityStudy:

    def __init__(self, device, dataset, model):
        self.device = device
        self.dataset = dataset
        self.model = model

    def create_reconstructions(self):
        reconstruction_dir = RECONSTRUCTION_DATA_DIR_FMT.format(self.dataset.cell_size, self.dataset.patch_size)
        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        for idx in tqdm(range(len(self.dataset)), total=len(self.dataset), desc='Creating reconstructions'):
            reconstruction = self.dataset.create_reconstructed_image(idx, add_grid=True)
            file_name = "reconstruction_{:d}_{:d}_{:}.png".format(self.dataset.cell_size, self.dataset.patch_size,
                                                                  self.dataset.bags_metadata[idx]['id'])
            reconstruction.save(reconstruction_dir + "/" + file_name)

    def sample_interpretations(self):
        random_idxs = list(range(len(self.dataset)))
        random.shuffle(random_idxs)

        for idx in random_idxs:
            bag, target = self.dataset[idx]
            self.create_interpretation(idx, bag, target)

    def create_interpretation_from_id(self, img_id):
        for idx, bag_md in enumerate(self.dataset.bags_metadata):
            if bag_md['id'] == img_id:
                bag, target = self.dataset[idx]
                self.create_interpretation(idx, bag, target)
                break
            else:
                continue

    def create_interpretation(self, idx, bag, target):
        bag_prediction, instance_predictions = self.model.forward_verbose(bag)
        print(bag_prediction)
        print(bag_prediction.shape)
        print(instance_predictions)
        print(instance_predictions.shape)
        print(torch.sum(instance_predictions, dim=0))

        max_abs_pred = max(abs(torch.min(instance_predictions).item()), abs(torch.max(instance_predictions).item()))
        norm = plt.Normalize(-max_abs_pred, max_abs_pred)
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["red", "lightgrey", self.dataset.target_to_rgb(clz)])
                 for clz in range(7)]

        reconstruction = self.dataset.create_reconstructed_image(idx, add_grid=False)
        original_mask = self.dataset.get_original_mask_img(idx)
        _, grid_ground_truth_coloured_mask = self._create_ground_truth_overall_mask(original_mask)
        masks = [self._create_instance_mask(instance_predictions, i) for i in range(7)]
        pred_masks = self._create_overall_mask(reconstruction, instance_predictions, cmaps, norm)
        _, _, overall_weighted_colour_mask, sat_img_with_overlay = pred_masks

        def format_axis(ax, title=None):
            ax.set_axis_off()
            ax.set_aspect('equal')
            if title is not None:
                ax.set_title(title)

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 7))
        axes[0][0].imshow(reconstruction)
        format_axis(axes[0][0], "Original Image")

        axes[0][1].imshow(original_mask)
        format_axis(axes[0][1], "True Overall Mask")

        axes[0][2].imshow(grid_ground_truth_coloured_mask)
        format_axis(axes[0][2], "True Overall Grid Mask")

        axes[0][3].imshow(overall_weighted_colour_mask)
        format_axis(axes[0][3], "Pred Overall Grid Mask")

        axes[1][0].imshow(sat_img_with_overlay)
        format_axis(axes[1][0], "Pred Overlay")

        for clz in range(7):
            axis = axes[(clz + 1) // 4 + 1][(clz + 1) % 4]
            im = axis.imshow(masks[clz], cmap=cmaps[clz], norm=norm)
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-max_abs_pred, 0, max_abs_pred])
            cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
            format_axis(axis, self.dataset.target_to_name(clz) + " pred")

        format_axis(axes[2][3])

        plt.tight_layout()
        plt.show()

    def _create_instance_mask(self, instance_predictions, clz):
        if self.dataset.grid_size == 153 and self.dataset.patch_size == 28:
            grid_dim = 16
        else:
            raise ValueError('Grid dimensionality not set for grid size {:d} and patch size {:d}'
                             .format(self.dataset.grid_size, self.dataset.patch_size))

        mask = np.zeros((grid_dim, grid_dim))
        for idx, instance_prediction in enumerate(instance_predictions[:, clz]):
            row_idx = idx // grid_dim
            col_idx = idx % grid_dim
            mask[row_idx, col_idx] = instance_prediction

        return mask

    def _create_overall_mask(self, sat_img, instance_predictions, cmaps, norm):
        if self.dataset.grid_size == 153 and self.dataset.patch_size == 28:
            grid_dim = 16
        else:
            raise ValueError('Grid dimensionality not set for grid size {:d} and patch size {:d}'
                             .format(self.dataset.grid_size, self.dataset.patch_size))

        overall_clz_mask = np.zeros((grid_dim, grid_dim))
        overall_colour_mask = np.zeros((grid_dim, grid_dim, 3))
        overall_weighted_colour_mask = np.zeros((grid_dim, grid_dim, 3))
        for idx, cell_instance_predictions in enumerate(instance_predictions):
            row_idx = idx // grid_dim
            col_idx = idx % grid_dim
            # Get the top predicted class for this cell
            top_clz = torch.argmax(cell_instance_predictions).item()

            # Mask by class values
            overall_clz_mask[row_idx, col_idx] = top_clz

            # Mask by colour
            rgb = self.dataset.target_to_rgb(top_clz)
            overall_colour_mask[row_idx, col_idx, 0] = rgb[0]
            overall_colour_mask[row_idx, col_idx, 1] = rgb[1]
            overall_colour_mask[row_idx, col_idx, 2] = rgb[2]

            # Mask by colour and weight
            top_pred_val = cell_instance_predictions[top_clz].detach().item()
            rgb = cmaps[top_clz](norm(top_pred_val))
            overall_weighted_colour_mask[row_idx, col_idx, 0] = rgb[0]
            overall_weighted_colour_mask[row_idx, col_idx, 1] = rgb[1]
            overall_weighted_colour_mask[row_idx, col_idx, 2] = rgb[2]

        overlay = Image.fromarray(np.uint8(overall_colour_mask * 255), mode='RGB')
        overlay = overlay.resize(sat_img.size, Image.NEAREST)
        overlay.putalpha(int(0.5 * 255))
        sat_img_with_overlay = sat_img.convert('RGBA')
        sat_img_with_overlay.paste(overlay, (0, 0), overlay)
        return overall_clz_mask, overall_colour_mask, overall_weighted_colour_mask, sat_img_with_overlay

    def _create_ground_truth_overall_mask(self, original_mask):
        grid_size = self.dataset.grid_size
        sat_img_arr = np.array(original_mask)

        # Iterate through each cell in the grid
        n_x = int(sat_img_arr.shape[0] / grid_size)
        n_y = int(sat_img_arr.shape[1] / grid_size)

        overall_clz_mask = np.zeros((n_x, n_y))
        overall_coloured_mask = np.zeros((n_x, n_y, 3))
        for i_x in range(n_x):
            for i_y in range(n_y):
                # Extract patch from original image
                p_x = i_x * grid_size
                p_y = i_y * grid_size
                patch_img_arr = sat_img_arr[p_x:p_x + grid_size, p_y:p_y + grid_size, :]

                # Get max colour in this patch and work out which class it is
                colours, counts = np.unique(patch_img_arr.reshape(-1, 3), axis=0, return_counts=1)
                top_idx = np.argmax(counts)
                top_colour = colours[top_idx]
                clz = self.dataset.rgb_to_target(top_colour[0] / 255, top_colour[1] / 255, top_colour[2] / 255)

                # Update masks
                overall_clz_mask[i_x, i_y] = clz
                overall_coloured_mask[i_x, i_y, 0] = top_colour[0]
                overall_coloured_mask[i_x, i_y, 1] = top_colour[1]
                overall_coloured_mask[i_x, i_y, 2] = top_colour[2]

        return overall_clz_mask, overall_coloured_mask
