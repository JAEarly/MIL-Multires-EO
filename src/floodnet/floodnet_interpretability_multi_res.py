import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
# Need to keep scienceplots imported for matplotlib styling even though the import is never used directly
# noinspection PyUnresolvedReferences
import scienceplots
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm

plt.style.use(['science', 'bright'])


nice_clz_names = ['Background', 'Building Flooded', 'Building Non-flooded', 'Road Flooded', 'Road Non-Flooded', 'Water',
                  'Tree', 'Vehicle', 'Pool', 'Grass']


class FloodNetMultiResInterpretabilityStudy:

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
            instance_targets = data['instance_targets']
            self.create_interpretation(idx, bag, target, metadata, instance_targets)

    def create_interpretation_from_id(self, img_id):
        print('Looking for image id: {:}'.format(img_id))
        for idx, bag_md in enumerate(self.dataset.bags_metadata):
            if bag_md['id'] == img_id:
                print(' Found')
                data = self.dataset[idx]
                bag = data['bag']
                target = data['target']
                metadata = data['bag_metadata']
                instance_targets = data['instance_targets']
                return self.create_interpretation(idx, bag, target, metadata, instance_targets)
            else:
                continue
        print(' Not found')

    def create_interpretation(self, idx, bag, target, metadata, instance_targets):
        orig_img = self.dataset.get_img(idx)
        mask_img = self.dataset.get_mask_img(idx)
        with torch.no_grad():
            bag_prediction, all_patch_preds = self.model.forward_verbose(bag, input_metadata=metadata)

        def format_axis(ax, title=None):
            ax.set_axis_off()
            ax.set_aspect('equal')
            if title is not None:
                ax.set_title(title, fontsize=16)

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 7))
        axes[0][0].imshow(orig_img)
        format_axis(axes[0][0], "Original Image")

        axes[0][1].imshow(np.array(mask_img) + 1e-6, cmap=self.dataset.cmap, vmin=0, vmax=10)
        format_axis(axes[0][1], "True Pixel Mask")

        true_s2_patch_mask = self._create_true_patch_mask(instance_targets[2])
        axes[0][2].imshow(np.array(true_s2_patch_mask) + 1e-6, cmap=self.dataset.cmap, vmin=0, vmax=10)
        format_axis(axes[0][2], "s = 2 True Patch Mask")

        for scale in range(4):
            patch_preds = all_patch_preds[scale].cpu()
            interpretation_out = self.create_interpretation_at_scale(idx, patch_preds)
            masks_out, max_abs_preds, norms, cmaps = interpretation_out
            pred_mask, clz_pred_masks, pred_weighted_mask, orig_img_with_overlay = masks_out

            axes[1][scale].imshow(pred_mask + 1e-6, cmap=self.dataset.cmap, vmin=0, vmax=10)
            format_axis(axes[1][scale], "s = {:} Predicted Mask".format(scale if scale < 3 else 'm'))

            if scale == 3:
                axes[0][3].imshow(orig_img_with_overlay)
                format_axis(axes[0][3], "s = m Predicted Overlay")
                clz_counts = Counter(pred_mask.flatten().tolist()).most_common()
                clz_order = [int(c[0]) for c in clz_counts]

                for order_idx in range(4):
                    axis = axes[2][order_idx]
                    if order_idx < len(clz_order):
                        clz = clz_order[order_idx]
                        im = axis.imshow(patch_preds[clz], cmap=cmaps[clz], norm=norms[clz])
                        divider = make_axes_locatable(axis)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                                            ticks=[-max_abs_preds[clz], 0, max_abs_preds[clz]])
                        cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
                        cbar.ax.tick_params(labelsize=14)
                        format_axis(axis, nice_clz_names[clz])
                    else:
                        format_axis(axis, '')

        plt.tight_layout()
        if self.show_outputs:
            plt.show()
        fig.savefig("out/interpretability/FloodNet/multi_res/{:d}_interpretability.png".format(idx), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def create_interpretation_at_scale(self, idx, patch_preds):
        # Work out max absolute prediction for each class, and create a normaliser for each class
        max_abs_preds = [max(abs(torch.min(patch_preds[c]).item()),
                             abs(torch.max(patch_preds[c]).item()))
                         for c in range(10)]
        norms = [plt.Normalize(-m, m) for m in max_abs_preds]
        # Create custom colour map for each class
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["black", "lightgrey", self.dataset.clz_idx_to_rgb(clz)])
                 for clz in range(10)]
        # Get predicted mask, predicted mask for each class, and a weighted mask
        orig_img = self.dataset.get_img(idx)
        masks_out = self._create_pred_masks(orig_img, patch_preds, cmaps, norms)
        return masks_out, max_abs_preds, norms, cmaps

    def _create_pred_masks(self, orig_img, patch_preds, cmaps, norms):
        _, grid_n_rows, grid_n_cols = patch_preds.shape

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
        pred_weighted_mask = Image.new('P', (grid_n_cols, grid_n_rows))
        pred_weighted_mask.putdata(overall_weight_mask_p.flatten())
        pred_weighted_mask.putpalette(weight_palette, rawmode="RGB")
        pred_weighted_mask = pred_weighted_mask.convert('RGB')

        overlay = pred_weighted_mask.resize(orig_img.size, Image.NEAREST)
        overlay.putalpha(int(0.75 * 255))
        orig_img_with_overlay = orig_img.convert('RGBA')
        orig_img_with_overlay.paste(overlay, (0, 0), overlay)

        return pred_mask, clz_pred_masks, pred_weighted_mask, orig_img_with_overlay

    def _create_true_patch_mask(self, instance_targets):
        return torch.max(instance_targets, dim=0)[1]
