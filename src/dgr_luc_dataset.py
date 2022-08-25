import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from overrides import overrides
from sklearn.model_selection import KFold, train_test_split
from torchvision import transforms
from tqdm import tqdm

from bonfire.data.mil_dataset import MilDataset
from bonfire.train.metrics import RegressionMetric, output_regression_results

RAW_DATA_DIR = 'data/DeepGlobeLUC/raw'
PATCH_DATA_DIR_FMT = 'data/DeepGlobeLUC/patch_{:d}_{:d}'
PATCH_DATA_CSV_FMT = 'data/DeepGlobeLUC/patch_{:d}_{:d}_data.csv'
TARGET_OUT_PATH = 'data/DeepGlobeLUC/targets.csv'
CLASS_DIST_PATH = 'data/DeepGlobeLUC/class_distribution.csv'


def setup():
    metadata_df = _load_metadata_df()
    _extract_patches(metadata_df)
    _calculate_dataset_normalisation(metadata_df)
    _visualise_data(metadata_df)
    _generate_per_class_coverage(metadata_df)
    _plot_per_class_coverage()
    _baseline_performance()


def _load_metadata_df():
    """
    Create a dataframe containing the metadata for each image (loaded from metadata.csv).
    """
    print('Loading image metadata')
    # Read data from csv
    metadata_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'metadata.csv'))
    # Discard anything that isn't train split (as other splits don't have segmentation ground truths
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    # Drop split column as we don't need it
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    # Update paths to images to be relative to project base
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    print(' Found {:d} images'.format(len(metadata_df)))
    return metadata_df


def _load_class_dict_df():
    print('Loading class data')
    class_dict_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'class_dict.csv'))
    class_dict_df[['r', 'g', 'b']] = class_dict_df[['r', 'g', 'b']].apply(lambda v: v // 255)
    class_dict_df['target'] = list(range(len(class_dict_df)))
    print(' Found {:d} classes'.format(len(class_dict_df)))
    return class_dict_df


def _extract_patches(metadata_df, grid_size=153, patch_size=28):
    """
    Extract patches from the training images.
    :param metadata_df: Dataframe of image metadata.
    :param grid_size: Size of each grid cell to extract from (original resolution)
    :param patch_size: Size to transform each patch to (new resolution)
    """
    # Calculate number of patches per original image (fixed original image size of 2448)
    num_patches = int(2448 / grid_size * 2448 / grid_size)
    # Calculate the relative reconstructed dimensionality of the new resolution image
    reconstructed_dim = int(num_patches ** 0.5 * patch_size)

    print('Extracting grid patches')
    print(' Grid size: {:d}'.format(grid_size))
    print(' Patch size: {:d}'.format(patch_size))
    print(' {:d} patches per image'.format(num_patches))
    print(' {:d} x {:d} effective new size'.format(reconstructed_dim, reconstructed_dim))

    # Create output directory or skip if output directory already exists (assumes previous extraction ran correctly)
    patch_dir = PATCH_DATA_DIR_FMT.format(grid_size, patch_size)
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    else:
        print(' Skipping...')
        return

    # Create a dataframe to map from image ids to a list of all the extracted patches for that image
    patches_df = metadata_df[['image_id']].copy()
    patches_df['patch_paths'] = ""

    # Loop over all the images
    for i in tqdm(range(len(metadata_df)), desc='Extracting patches', leave=False):
        # Load original satellite image
        image_id = metadata_df['image_id'][i]
        sat_path = metadata_df['sat_image_path'][i]
        # Load image and resize to new target resolution
        sat_img = Image.open(sat_path)
        sat_img.thumbnail((reconstructed_dim, reconstructed_dim))
        sat_img_arr = np.array(sat_img)
        # Iterate through each cell in the grid
        n_x = int(sat_img_arr.shape[0]/patch_size)
        n_y = int(sat_img_arr.shape[1]/patch_size)
        patch_paths = []
        for i_x in range(n_x):
            for i_y in range(n_y):
                # Extract patch from original image
                p_x = i_x * patch_size
                p_y = i_y * patch_size
                patch_img_arr = sat_img_arr[p_x:p_x+patch_size, p_y:p_y+patch_size, :]
                patch_path = "{:s}/{:d}_{:d}_{:d}.png".format(patch_dir, image_id, i_x, i_y)
                patch_paths.append(patch_path)
                patch_img = Image.fromarray(patch_img_arr)
                patch_img.save(patch_path)
        # Add list of patch paths for this image to the dataframe
        patches_df.loc[i, 'patch_paths'] = ",".join(patch_paths)
    # Save the patch dataframe
    patches_df.to_csv(PATCH_DATA_CSV_FMT.format(grid_size, patch_size), index=False)


def _calculate_dataset_normalisation(metadata_df):
    print('Calculating dataset normalisation')
    avgs = []
    for i in tqdm(range(len(metadata_df)), desc='Calculating dataset normalisation', leave=False):
        sat_path = metadata_df['sat_image_path'][i]
        sat_img = Image.open(sat_path)
        sat_img_arr = np.array(sat_img) / 255
        avg = np.mean(sat_img_arr, axis=(0, 1))
        avgs.append(avg)
    arrs = np.stack(avgs)
    arrs_mean = np.mean(arrs, axis=0)
    arrs_std = np.std(arrs, axis=0)
    print(' Mean:', arrs_mean)
    print('  Std:', arrs_std)


def _make_binary_mask(mask):
    binary_mask = torch.zeros_like(torch.as_tensor(mask))
    binary_mask[mask > 128] = 1
    return binary_mask


def _make_single_target_mask(mask_binary, target_clz):
    # mask_img should already be binary
    assert mask_binary.min() >= 0
    assert mask_binary.max() <= 1
    rgb = DgrLucDataset.target_to_rgb(target_clz)
    c1 = mask_binary[:, :, 0] == rgb[0]
    c2 = mask_binary[:, :, 1] == rgb[1]
    c3 = mask_binary[:, :, 2] == rgb[2]
    new_mask = (c1 & c2 & c3)
    return new_mask


def _visualise_data(metadata_df, n_to_show=5):
    random_idxs = np.random.choice(len(metadata_df), size=n_to_show, replace=False)
    for i in random_idxs:
        sat_path = metadata_df['sat_image_path'][i]
        mask_path = metadata_df['mask_path'][i]
        sat_img = Image.open(sat_path)
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        mask_binary = _make_binary_mask(mask_arr)

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        axes[0][0].imshow(sat_img, vmin=0, vmax=255)
        axes[0][0].set_title("Satellite Image")
        axes[0][1].imshow(mask_binary.float(), vmin=0, vmax=1)
        axes[0][1].set_title("Complete Mask")
        for target in range(7):
            single_mask = _make_single_target_mask(mask_binary, target)
            axes[(target + 2) // 3][(target + 2) % 3].imshow(single_mask, vmin=0, vmax=1, cmap='gray')
            axes[(target + 2) // 3][(target + 2) % 3].set_title(DgrLucDataset.target_to_name(target))
        plt.tight_layout()
        plt.show()


def _generate_per_class_coverage(metadata_df):
    print('Generating class distribution')
    if os.path.exists(CLASS_DIST_PATH):
        print(' Skipping...')
        return

    cover_dist_df = metadata_df[['image_id']].copy()
    for i in range(7):
        cover_dist_df[DgrLucDataset.target_to_name(i)] = pd.Series(dtype=float)

    for i in tqdm(range(len(metadata_df)), desc='Calculating class coverage for each image', leave=False):
        mask_path = metadata_df['mask_path'][i]
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        mask_binary = _make_binary_mask(mask_arr)

        s = 0
        for target in range(7):
            single_mask = _make_single_target_mask(mask_binary, target)
            name = DgrLucDataset.target_to_name(target)
            percentage_cover = len(single_mask.nonzero())/single_mask.numel()
            cover_dist_df.loc[i, name] = percentage_cover
            s += percentage_cover
        assert abs(s - 1) < 1e-6
    cover_dist_df.to_csv(CLASS_DIST_PATH, index=False)


def _plot_per_class_coverage():
    cover_dist_df = DgrLucDataset.load_per_class_coverage()
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(14, 2))
    for target in range(7):
        name = DgrLucDataset.target_to_name(target)
        dist = cover_dist_df[name]
        axes[target].set_title(DgrLucDataset.target_to_name(target))
        axes[target].hist(dist, bins=25, range=(0, 1), log=True)
        axes[target].set_xlabel('Coverage')
        axes[target].set_ylabel('Density')
        axes[target].set_ylim(top=1000)
    plt.tight_layout()
    fig.savefig(CLASS_DIST_PATH.replace("csv", "png"), format='png', dpi=300)
    plt.show()


def _baseline_performance():

    def performance_for_dataset(pred, dataset):
        targets = dataset.targets
        preds = torch.ones_like(targets)
        preds *= pred
        results = RegressionMetric.calculate_metric(preds, targets, None)
        return results

    idx = 0
    results_arr = np.full((1, 5, 3), np.nan, dtype=object)
    for train_dataset, val_dataset, test_dataset in DgrLucDataset.create_datasets():
        train_mean_target = train_dataset.targets.mean(dim=0)
        train_results = performance_for_dataset(train_mean_target, train_dataset)
        val_results = performance_for_dataset(train_mean_target, val_dataset)
        test_results = performance_for_dataset(train_mean_target, test_dataset)
        results_arr[:, idx, :] = [train_results, val_results, test_results]
        idx += 1
    output_regression_results(['Baseline'], results_arr)


class DgrLucDataset(MilDataset):

    name = 'dgr_luc'
    d_in = 1200
    n_expected_dims = 4  # i x c x h x w
    n_classes = 7
    metric_clz = RegressionMetric
    class_dict_df = _load_class_dict_df()
    dataset_mean = (0.4082, 0.3791, 0.2816)
    dataset_std = (0.06722, 0.04668, 0.04768)
    basic_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)])

    def __init__(self, bags, targets, bags_metadata):
        super().__init__(bags, targets, None, bags_metadata)
        self.transform = self.basic_transform

    @classmethod
    def load_dgr_bags(cls, grid_size=153, patch_size=28):
        patches_df = pd.read_csv(PATCH_DATA_CSV_FMT.format(grid_size, patch_size))
        coverage_df = cls.load_per_class_coverage()
        complete_df = pd.merge(patches_df, coverage_df, on='image_id')
        bags = np.asarray([s.split(",") for s in complete_df['patch_paths'].tolist()])
        targets = complete_df[cls.get_clz_names()].to_numpy()
        bags_metadata = np.asarray([{'id': id_} for id_ in complete_df['image_id'].tolist()])
        return bags, targets, bags_metadata

    @classmethod
    def target_to_rgb(cls, target):
        r = cls.class_dict_df.loc[cls.class_dict_df['target'] == target]
        rgb = r[['r', 'g', 'b']].values.tolist()[0]
        return rgb

    @classmethod
    def target_to_name(cls, target):
        return cls.class_dict_df['name'][target]

    @staticmethod
    def load_per_class_coverage():
        return pd.read_csv(CLASS_DIST_PATH)

    @classmethod
    def create_datasets(cls, random_state=12, grid_size=153, patch_size=28):
        bags, targets, bags_metadata = DgrLucDataset.load_dgr_bags(patch_size=patch_size)

        for train_split, val_split, test_split in cls.get_dataset_splits(bags, targets, random_state=random_state):
            # Setup bags, targets, and metadata for splits
            train_bags, val_bags, test_bags = [bags[i] for i in train_split], \
                                              [bags[i] for i in val_split], \
                                              [bags[i] for i in test_split]
            train_targets, val_targets, test_targets = targets[train_split], targets[val_split], targets[test_split]
            train_md, val_md, test_md = bags_metadata[train_split], bags_metadata[val_split], bags_metadata[test_split]

            train_dataset = DgrLucDataset(train_bags, train_targets, train_md)
            val_dataset = DgrLucDataset(val_bags, val_targets, val_md)
            test_dataset = DgrLucDataset(test_bags, test_targets, test_md)

            yield train_dataset, val_dataset, test_dataset

    @classmethod
    def get_dataset_splits(cls, bags, targets, random_state=5):
        # Split using stratified k fold (5 splits)
        skf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = skf.split(bags, targets)

        # Split further into train/val/test (80/10/10)
        for train_split, test_split in splits:

            # Split val split (currently 20% of data) into 10% and 10% (so 50/50 ratio)
            val_split, test_split = train_test_split(test_split, random_state=random_state, test_size=0.5)
            # Yield splits
            yield train_split, val_split, test_split

    # TODO should this be a property?
    @classmethod
    def get_clz_names(cls):
        return cls.class_dict_df['name'].tolist()

    @classmethod
    def create_complete_dataset(cls):
        raise NotImplementedError

    @classmethod
    def get_target_mask(cls, instance_targets, clz):
        pass

    @overrides
    def summarise(self, out_clz_dist=True):
        print('- MIL Dataset Summary -')
        print(' {:d} bags'.format(len(self.bags)))

        if out_clz_dist:
            print(' Class Distribution')
            for clz in range(self.n_classes):
                print('  Class {:d} - {:s}'.format(clz, self.get_clz_names()[clz]))
                clz_targets = self.targets[:, clz]
                hist, bins = np.histogram(clz_targets, bins=np.linspace(0, 1, 11))
                for i in range(len(hist)):
                    print('   {:.1f}-{:.1f}: {:d}'.format(bins[i], bins[i + 1], hist[i]))

        bag_sizes = [len(b) for b in self.bags]
        print(' Bag Sizes')
        print('  Min: {:d}'.format(min(bag_sizes)))
        print('  Avg: {:.1f}'.format(np.mean(bag_sizes)))
        print('  Max: {:d}'.format(max(bag_sizes)))

    def __getitem__(self, bag_idx):
        instances = []
        bag = self.bags[bag_idx]
        for patch_path in bag:
            instance = Image.open(patch_path)
            if self.transform is not None:
                instance = self.transform(instance)
            instances.append(instance)
        instances = torch.stack(instances)
        target = self.targets[bag_idx]
        return instances, target


if __name__ == "__main__":
    setup()
