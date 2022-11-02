import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dgr_luc_interpretability import MilLucInterpretabilityStudy


class UnetLucInterpretabilityStudy(MilLucInterpretabilityStudy):

    def create_interpretation(self, idx, bag, target, bag_id):
        save_path = "out/interpretability/{:}/{:}_interpretation.png".format(self.dataset.model_type, bag_id)
        if not self.show_outputs and os.path.exists(save_path):
            return

        bag_prediction, instance_predictions = self.model.forward_verbose(bag)
        # print('  Pred:', ['{:.3f}'.format(p) for p in bag_prediction])
        # print('Target:', ['{:.3f}'.format(t) for t in target])

        # Work out max absolute prediction for each class, and create a normaliser for each class
        max_abs_preds = [max(abs(torch.min(instance_predictions[c]).item()),
                             abs(torch.max(instance_predictions[c]).item()))
                         for c in range(7)]
        norms = [plt.Normalize(-m, m) for m in max_abs_preds]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",
                                                              ["red", "lightgrey", self.dataset.target_to_rgb(clz)])
                 for clz in range(7)]

        sat_img = self.dataset.get_sat_img(idx)
        mask_img = self.dataset.get_mask_img(idx)
        _, grid_ground_truth_coloured_mask = self._create_ground_truth_overall_mask(mask_img)
        masks = [self._create_instance_mask(instance_predictions, i) for i in range(7)]
        pred_masks = self._create_overall_mask(sat_img, instance_predictions, cmaps, norms)
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

        axes[0][2].imshow(overall_colour_mask)
        format_axis(axes[0][2], "Prediction Mask")

        axes[0][3].imshow(overall_weighted_colour_mask)
        format_axis(axes[0][3], "Weighted Pred. Mask")

        for clz_idx in range(4):
            axis = axes[1][clz_idx]
            if clz_idx < len(clz_order):
                clz = clz_order[clz_idx]
                im = axis.imshow(masks[clz], cmap=cmaps[clz], norm=norms[clz])
                divider = make_axes_locatable(axis)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                                    ticks=[-max_abs_preds[clz], 0, max_abs_preds[clz]])
                cbar.ax.set_yticklabels(['-ve', '0', '+ve'])
                cbar.ax.tick_params(labelsize=14)
                format_axis(axis, self.dataset.target_to_name(clz).replace('_', ' ').title())
            else:
                format_axis(axis, '')

        plt.tight_layout()
        if self.show_outputs:
            plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def _create_instance_mask(self, instance_predictions, clz):
        return instance_predictions[clz].cpu().detach().numpy()

    def _create_overall_mask(self, sat_img, instance_predictions, cmaps, norms):
        _, grid_size, _ = instance_predictions.shape

        # Create mask of top clz and its weight for each pixel
        overall_weight_mask, overall_clz_mask = torch.max(instance_predictions, dim=0)
        overall_weight_mask = overall_weight_mask.detach().cpu()
        overall_clz_mask = overall_clz_mask.detach().cpu()

        # Create palette to map from clz to colours
        clz_palette = self.dataset.create_clz_palette()

        # Create color mask from palette and clz mask
        overall_colour_mask = Image.new('P', (grid_size, grid_size))
        overall_colour_mask.putdata(torch.flatten(overall_clz_mask).numpy())
        overall_colour_mask.putpalette(clz_palette, rawmode="RGB")
        overall_colour_mask = overall_colour_mask.convert('RGB')

        # Create weight palette
        #  Each class is mapped to a different range
        #    Clz 0 -> 0 to 35
        #    Clz 1 -> 36 to 71
        #    etc.
        #  Each class range is then mapped to it's colour map
        #  Max length of a palette is 768 (256 RGB colours), so with 7 classes, the max range is 36 (36 * 7 * 3 = 756)
        weight_palette = []
        for clz in range(7):
            cmap = cmaps[clz]
            for i in range(36):
                val = cmap(i / 36)
                color = [int(c * 255) for c in val[:3]]
                weight_palette.extend(color)

        # Normalise weight mask
        #  Do for all classes first, then chose the correct normed value based on the selected class for each pixel
        norm_overall_weight_masks = [norm(overall_weight_mask) for norm in norms]
        norm_overall_weight_mask = np.zeros_like(overall_weight_mask)
        for clz in range(7):
            norm_overall_weight_mask = np.where(overall_clz_mask == clz,
                                                norm_overall_weight_masks[clz],
                                                norm_overall_weight_mask)

        # Convert the weight mask to match the weight palette values
        #  Normalise
        #  Convert to range (0 to 35) and round to nearest int
        #  Add to clz mask values (multiplied by 36 to map to range start values)
        overall_weight_mask_p = (overall_clz_mask * 36).numpy() + np.floor(norm_overall_weight_mask * 36).astype(int)

        # Create weighted color mask from palette and clz mask
        overall_weighted_colour_mask = Image.new('P', (grid_size, grid_size))
        overall_weighted_colour_mask.putdata(overall_weight_mask_p.flatten())
        overall_weighted_colour_mask.putpalette(weight_palette, rawmode="RGB")
        overall_weighted_colour_mask = overall_weighted_colour_mask.convert('RGB')

        overlay = overall_colour_mask.resize(sat_img.size, Image.NEAREST)
        overlay.putalpha(int(0.5 * 255))
        sat_img_with_overlay = sat_img.convert('RGBA')
        sat_img_with_overlay.paste(overlay, (0, 0), overlay)

        return overall_clz_mask, overall_colour_mask, overall_weighted_colour_mask, sat_img_with_overlay
