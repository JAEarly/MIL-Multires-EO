import os
import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


class FloodNetInterpretabilityStudy:

    def __init__(self, device, dataset, model, show_outputs):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.show_outputs = show_outputs

    def sample_interpretations(self):
        random_idxs = list(range(len(self.dataset)))
        random.shuffle(random_idxs)

        for idx in tqdm(random_idxs, 'Generating interpretations'):
            data = self.dataset[idx]
            bag = data['bag']
            target = data['target']
            metadata = data['bag_metadata']
            self.create_interpretation(idx, bag, target, metadata)

    def create_interpretation_from_id(self, img_id):
        print('Looking for image id: {:}'.format(img_id))
        for idx, bag_md in enumerate(self.dataset.bags_metadata):
            if bag_md['id'] == img_id:
                print(' Found')
                data = self.dataset[idx]
                bag = data['bag']
                target = data['target']
                metadata = data['bag_metadata']
                return self.create_interpretation(idx, bag, target, metadata)
            else:
                continue
        print(' Not found')

    def create_interpretation(self, idx, bag, target, metadata):
        save_path = "out/interpretability/{:}/{:}_interpretation.png".format(self.dataset.model_type, metadata['id'])

        print(bag)
        print(metadata)
        bag_prediction, patch_preds = self.model.forward_verbose(bag, input_metadata=metadata)
        patch_preds = patch_preds.detach().cpu()
        # print('  Pred:', ['{:.3f}'.format(p) for p in bag_prediction])
        # print('Target:', ['{:.3f}'.format(t) for t in target])

        # Work out max absolute prediction for each class, and create a normaliser for each class
        max_abs_preds = [max(abs(torch.min(patch_preds[c]).item()),
                             abs(torch.max(patch_preds[c]).item()))
                         for c in range(10)]
        norms = [plt.Normalize(-m, m) for m in max_abs_preds]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["black", "lightgrey", self.dataset.clz_idx_to_rgb(clz)])
                 for clz in range(10)]

        orig_img = self.dataset.get_img(idx)
        mask_img = self.dataset.get_mask_img(idx)
        mask_img = mask_img.resize((self.dataset.patch_details.effective_cell_resolution_x,
                                    self.dataset.patch_details.effective_cell_resolution_y),
                                   resample=Image.Resampling.NEAREST)

        # _, grid_ground_truth_coloured_mask = self._create_ground_truth_overall_mask(mask_img)
        # pred_masks = self._create_overall_mask(sat_img, patch_preds, cmaps, norms)
        # pred_clz_mask, overall_colour_mask, overall_weighted_colour_mask, _ = pred_masks

        pred_mask, clz_pred_masks, pred_weighted_mask = self._create_pred_masks(orig_img, patch_preds, cmaps, norms)

        # Can skip plotting if we're not showing the output and the file already exists
        if self.show_outputs or not os.path.exists(save_path):
            clz_counts = Counter(pred_mask.flatten().tolist()).most_common()
            clz_order = [int(c[0]) for c in clz_counts]

            def format_axis(ax, title=None):
                ax.set_axis_off()
                ax.set_aspect('equal')
                if title is not None:
                    ax.set_title(title, fontsize=16)

            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))
            axes[0][0].imshow(orig_img)
            format_axis(axes[0][0], "Original Image")

            axes[0][1].imshow(np.array(mask_img) + 1e-6, cmap='tab10', vmin=0, vmax=10)
            format_axis(axes[0][1], "True Pixel Mask")

            axes[0][2].imshow(self._create_true_patch_mask(mask_img).T + 1e-6, cmap='tab10', vmin=0, vmax=10)
            format_axis(axes[0][2], "True Patch Mask")

            axes[0][3].imshow(pred_mask.T + 1e-6, cmap='tab10', vmin=0, vmax=10)
            format_axis(axes[0][3], "Predicted Mask")

            axes[0][4].imshow(pred_weighted_mask)
            format_axis(axes[0][4], "W Pred Mask")

            for order_idx in range(5):
                axis = axes[1][order_idx]
                if order_idx < len(clz_order):
                    clz = clz_order[order_idx]
                    im = axis.imshow(patch_preds[clz].T, cmap=cmaps[clz], norm=norms[clz])
                    # divider = make_axes_locatable(axis)
                    # cax = divider.append_axes('right', size='5%', pad=0.05)
                    # cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                    #                     ticks=[-max_abs_preds[clz], 0, max_abs_preds[clz]])
                    # cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
                    # cbar.ax.tick_params(labelsize=14)
                    format_axis(axis, self.dataset.clz_names[clz])
                else:
                    format_axis(axis, '')

            plt.tight_layout()
            if self.show_outputs:
                plt.show()
            # fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

        grid_ground_truth_coloured_mask = None
        return orig_img, mask_img, grid_ground_truth_coloured_mask, None, None

    def _create_pred_masks(self, orig_img, patch_preds, cmaps, norms):
        _, grid_size_x, grid_size_y = patch_preds.shape

        # Create mask of top clz, clz weights for each pixel, and clz pred masks
        weight_mask, pred_mask = torch.max(patch_preds, dim=0)
        clz_pred_masks = [norm(weight_mask) for norm in norms]

        # Create weight palette
        #  Each class is mapped to a different range
        #    Clz 0 -> 0 to 24
        #    Clz 1 -> 25 to 49
        #    etc.
        #  Each class range is then mapped to it's colour map
        #  Max length of a palette is 768 (256 RGB colours), so with 10 classes, the max range is 25 (25 * 10 * 3 = 750)
        weight_palette = []
        for clz in range(10):
            cmap = cmaps[clz]
            for i in range(25):
                val = cmap(i / 25)
                color = [int(c * 255) for c in val[:3]]
                weight_palette.extend(color)

        # Normalise weight mask
        #  Do for all classes first, then chose the correct normed value based on the selected class for each pixel
        overall_weight_mask = np.zeros_like(weight_mask)
        for clz in range(10):
            overall_weight_mask = np.where(pred_mask == clz,
                                           clz_pred_masks[clz],
                                           overall_weight_mask)

        # Convert the weight mask to match the weight palette values
        #  Convert to range (0 to 24) and round to nearest int
        rounded_norm_overall_weight_mask = np.floor(overall_weight_mask * 24).astype(int)
        #  Add clz mask values (multiplied by 25 to map to range start values)
        overall_weight_mask_p = rounded_norm_overall_weight_mask + (pred_mask * 25).numpy()

        # Create weighted color mask from palette and clz mask
        pred_weighted_mask = Image.new('P', (grid_size_x, grid_size_y))
        pred_weighted_mask.putdata(overall_weight_mask_p.T.flatten())
        pred_weighted_mask.putpalette(weight_palette, rawmode="RGB")
        pred_weighted_mask = pred_weighted_mask.convert('RGB')

        # overlay = overall_colour_mask.resize(sat_img.size, Image.NEAREST)
        # overlay.putalpha(int(0.5 * 255))
        # sat_img_with_overlay = sat_img.convert('RGBA')
        # sat_img_with_overlay.paste(overlay, (0, 0), overlay)

        return pred_mask, clz_pred_masks, pred_weighted_mask

    def _create_true_patch_mask(self, mask_img):
        grid_size_x = self.dataset.patch_details.grid_size_x
        grid_size_y = self.dataset.patch_details.grid_size_y
        cell_size_x = self.dataset.patch_details.cell_size_x
        cell_size_y = self.dataset.patch_details.cell_size_y

        print(mask_img.width, mask_img.height)
        mask_img_arr = np.array(mask_img).T

        print(mask_img_arr.shape)

        overall_clz_mask = np.zeros((grid_size_x, grid_size_y))
        for i_x in range(grid_size_x):
            for i_y in range(grid_size_y):
                # Extract patch from original image
                p_x = i_x * cell_size_x
                p_y = i_y * cell_size_y
                patch_mask_arr = mask_img_arr[p_x:p_x + cell_size_x, p_y:p_y + cell_size_y]

                # Get max colour in this patch and work out which class it is
                clzs, counts = np.unique(patch_mask_arr, return_counts=True)
                top_idx = np.argmax(counts)

                # Update masks
                overall_clz_mask[i_x, i_y] = clzs[top_idx]

        return overall_clz_mask
