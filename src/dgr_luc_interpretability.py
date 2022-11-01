import os
import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from dgr_luc_dataset import RECONSTRUCTION_DATA_DIR_FMT

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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

        for idx in tqdm(random_idxs, 'Generating interpretations'):
            data = self.dataset[idx]
            bmd = self.dataset.bags_metadata[idx]
            bag, target = data[0], data[1]
            self.create_interpretation(idx, bag, target, bmd['id'])

    def create_interpretation_from_id(self, img_id):
        print('Looking for image id: {:}'.format(img_id))
        for idx, bag_md in enumerate(self.dataset.bags_metadata):
            if bag_md['id'] == img_id:
                print(' Found')
                data = self.dataset[idx]
                bmd = self.dataset.bags_metadata[idx]
                bag, target = data[0], data[1]
                self.create_interpretation(idx, bag, target, bmd['id'])
                exit(0)
            else:
                continue
        print(' Not found')

    def create_interpretation(self, idx, bag, target, bag_id):
        save_path = "out/interpretability/{:}/{:}_interpretation.png".format(self.dataset.model_type, bag_id)
        if os.path.exists(save_path):
            return

        bag_prediction, instance_predictions = self.model.forward_verbose(bag)
        # print('  Pred:', ['{:.3f}'.format(p) for p in bag_prediction])
        # print('Target:', ['{:.3f}'.format(t) for t in target])

        max_abs_pred = max(abs(torch.min(instance_predictions).item()), abs(torch.max(instance_predictions).item()))
        norm = plt.Normalize(-max_abs_pred, max_abs_pred)
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["red", "lightgrey", self.dataset.target_to_rgb(clz)])
                 for clz in range(7)]

        sat_img = self.dataset.get_sat_img(idx)
        mask_img = self.dataset.get_mask_img(idx)
        _, grid_ground_truth_coloured_mask = self._create_ground_truth_overall_mask(mask_img)
        masks = [self._create_instance_mask(instance_predictions, i) for i in range(7)]
        pred_masks = self._create_overall_mask(sat_img, instance_predictions, cmaps, norm)
        pred_clz_mask, overall_colour_mask, overall_weighted_colour_mask, sat_img_with_overlay = pred_masks

        clz_counts = Counter(pred_clz_mask.flatten().tolist()).most_common()
        clz_order = [int(c[0]) for c in clz_counts]

        def format_axis(ax, title=None):
            ax.set_axis_off()
            ax.set_aspect('equal')
            if title is not None:
                ax.set_title(title, fontsize=16)

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5))
        axes[0][0].imshow(sat_img)
        format_axis(axes[0][0], "Original Image")

        axes[0][1].imshow(mask_img)
        format_axis(axes[0][1], "Original Mask")

        axes[0][2].imshow(grid_ground_truth_coloured_mask)
        format_axis(axes[0][2], "Grid Mask")

        axes[0][3].imshow(overall_weighted_colour_mask)
        format_axis(axes[0][3], "Predicted Mask")

        for clz_idx in range(4):
            axis = axes[1][clz_idx]
            if clz_idx < len(clz_order):
                clz = clz_order[clz_idx]
                im = axis.imshow(masks[clz], cmap=cmaps[clz], norm=norm)
                divider = make_axes_locatable(axis)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-max_abs_pred, 0, max_abs_pred])
                cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
                cbar.ax.tick_params(labelsize=14)
                format_axis(axis, self.dataset.target_to_name(clz).replace('_', ' ').title())
            else:
                format_axis(axis, '')

        plt.tight_layout()
        # plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def _create_instance_mask(self, instance_predictions, clz):
        grid_size = self.dataset.patch_details.grid_size

        mask = np.zeros((grid_size, grid_size))
        for idx, instance_prediction in enumerate(instance_predictions[:, clz]):
            row_idx = idx // grid_size
            col_idx = idx % grid_size
            mask[row_idx, col_idx] = instance_prediction

        return mask

    def _create_overall_mask(self, sat_img, instance_predictions, cmaps, norm):
        grid_size = self.dataset.patch_details.grid_size

        overall_clz_mask = np.zeros((grid_size, grid_size))
        overall_colour_mask = np.zeros((grid_size, grid_size, 3))
        overall_weighted_colour_mask = np.zeros((grid_size, grid_size, 3))
        for idx, cell_instance_predictions in enumerate(instance_predictions):
            row_idx = idx // grid_size
            col_idx = idx % grid_size
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
        grid_size = self.dataset.patch_details.grid_size
        cell_size = self.dataset.patch_details.cell_size
        sat_img_arr = np.array(original_mask)

        overall_clz_mask = np.zeros((grid_size, grid_size))
        overall_coloured_mask = np.zeros((grid_size, grid_size, 3))
        for i_x in range(grid_size):
            for i_y in range(grid_size):
                # Extract patch from original image
                p_x = i_x * cell_size
                p_y = i_y * cell_size
                patch_img_arr = sat_img_arr[p_x:p_x + cell_size, p_y:p_y + cell_size, :]

                # Get max colour in this patch and work out which class it is
                colours, counts = np.unique(patch_img_arr.reshape(-1, 3), axis=0, return_counts=1)
                top_idx = np.argmax(counts)
                top_colour = colours[top_idx]
                clz = self.dataset.rgb_to_target(top_colour[0] / 255, top_colour[1] / 255, top_colour[2] / 255)

                # Update masks
                overall_clz_mask[i_x, i_y] = clz
                overall_coloured_mask[i_x, i_y, 0] = top_colour[0] / 255
                overall_coloured_mask[i_x, i_y, 1] = top_colour[1] / 255
                overall_coloured_mask[i_x, i_y, 2] = top_colour[2] / 255

        return overall_clz_mask, overall_coloured_mask
