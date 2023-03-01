import os

import numpy as np
import pandas as pd
# Need to keep scienceplots imported for matplotlib styling even though the import is never used directly
# noinspection PyUnresolvedReferences
import scienceplots
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from deepglobe.dgr_luc_dataset import _load_metadata_df as load_deepglobe_metadata, DgrLucDataset, make_binary_mask, \
    _make_single_target_mask
from floodnet.floodnet_dataset import _load_metadata_df as load_floodnet_metadata, FloodNetDataset
import matplotlib as mpl
plt.style.use(['science', 'bright'])


def run():
    print('Running for DeepGlobe...')
    deepglobe_cover_df = get_deepglobe_distribution()
    deepglobe_nice_names = ['Urban', 'Agricultural', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
    print('Plotting...')
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(14, 2))
    for target in range(7):
        _plot_per_class_coverage(deepglobe_cover_df[DgrLucDataset.target_to_name(target)], axes[target],
                                 deepglobe_nice_names[target], 1000)
    axes[3].set_xlabel('Coverage', size=14)
    axes[0].set_ylabel('Density', size=15)
    axes[0].set_yticklabels(["0", "$10^0$", "$10^1$", "$10^2$", "$10^3$"], size=12)
    plt.tight_layout()
    fig.savefig("out/fig/class_dist_DeepGlobe.png", format='png', dpi=300)
    plt.show()

    print('Running for FloodNet...')
    floodnet_cover_df = get_floodnet_distribution()
    floodnet_nice_names = ['Background', 'Building Flooded', 'Building Non-flooded', 'Road Flooded', 'Road Non-Flooded',
                           'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
    print('Plotting...')
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 4))
    for target in range(10):
        _plot_per_class_coverage(floodnet_cover_df[FloodNetDataset.label_to_name(target)],
                                 axes[target // 5][target % 5], floodnet_nice_names[target], 3500)
    axes[1][2].set_xlabel('Coverage', size=14)
    axes[0][0].set_ylabel('Density', size=15)
    axes[1][0].set_ylabel('Density', size=15)
    axes[0][0].set_yticklabels(["0", "$10^0$", "$10^1$", "$10^2$", "$10^3$"], size=12)
    axes[1][0].set_yticklabels(["0", "$10^0$", "$10^1$", "$10^2$", "$10^3$"], size=12)
    plt.tight_layout()
    fig.savefig("out/fig/class_dist_FloodNet.png", format='png', dpi=300)
    plt.show()


def get_deepglobe_distribution():
    metadata_df = load_deepglobe_metadata()
    class_dist_path = "data/DeepGlobeLUC/class_distribution.csv"

    if os.path.exists(class_dist_path):
        print('Loading class distribution')
        cover_dist_df = pd.read_csv(class_dist_path)
    else:
        print('Generating class distribution')
        cover_dist_df = metadata_df[['image_id']].copy()
        for i in range(7):
            cover_dist_df[DgrLucDataset.target_to_name(i)] = pd.Series(dtype=float)

        for i in tqdm(range(len(metadata_df)), desc='Calculating class coverage for each image', leave=False):
            mask_path = metadata_df['mask_path'][i]
            mask_img = Image.open(mask_path)
            mask_arr = np.array(mask_img)
            mask_binary = make_binary_mask(mask_arr)
            s = 0
            for target in range(7):
                single_mask = _make_single_target_mask(mask_binary, target)
                name = DgrLucDataset.target_to_name(target)
                percentage_cover = len(single_mask.nonzero()) / single_mask.numel()
                cover_dist_df.loc[i, name] = percentage_cover
                s += percentage_cover
            assert abs(s - 1) < 1e-6
        cover_dist_df.to_csv(class_dist_path, index=False)
    return cover_dist_df


def get_floodnet_distribution():
    metadata_df = load_floodnet_metadata()
    class_dist_path = "data/FloodNet/class_distribution.csv"
    if os.path.exists(class_dist_path):
        print('Loading class distribution')
        cover_dist_df = pd.read_csv(class_dist_path)
    else:
        print('Generating class distribution')
        cover_dist_df = metadata_df[['image_id']].copy()
        for i in range(10):
            cover_dist_df[FloodNetDataset.label_to_name(i)] = pd.Series(dtype=float)
        for i in tqdm(range(len(metadata_df)), desc='Calculating class coverage for each image', leave=False):
            mask_path = metadata_df['mask_path'][i]
            mask_img = Image.open(mask_path)
            mask_arr = np.array(mask_img)
            img_targets, img_counts = np.unique(mask_arr, return_counts=True)
            s = 0
            for target in range(10):
                name = FloodNetDataset.label_to_name(target)
                if target in img_targets:
                    percentage_coverage = img_counts[img_targets.tolist().index(target)] / np.sum(img_counts)
                else:
                    percentage_coverage = 0
                cover_dist_df.loc[i, name] = percentage_coverage
                s += percentage_coverage
            assert abs(s - 1) < 1e-6
        cover_dist_df.to_csv(class_dist_path, index=False)
    return cover_dist_df


def _plot_per_class_coverage(dist, axis, nice_clz_name, y_top):
    axis.set_title(nice_clz_name, size=14)
    axis.hist(dist, bins=25, range=(0, 1))

    # General tick formatting
    axis.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=12)
    axis.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelsize=12)
    axis.set_xlim(right=1)

    # Configure log axis
    axis.set_yscale('log')
    axis.set_ylim(0.1, y_top)
    axis.set_yticks([0.1, 1, 10, 100, 1000])
    axis.set_yticklabels([])

    # Add minor log ticks
    minor_loc = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    axis.yaxis.set_minor_locator(minor_loc)
    axis.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


if __name__ == "__main__":
    run()
