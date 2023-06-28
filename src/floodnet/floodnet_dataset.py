import os
from abc import ABC
from abc import abstractmethod
from pathlib import PurePosixPath

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from overrides import overrides
from torchvision import transforms
from tqdm import tqdm

from bonfire.data.mil_dataset import MilDataset
from bonfire.train.metrics import RegressionMetric
from dataset import PatchDetails

BASE_DATA_DIR = PurePosixPath('data/FloodNet')
METADATA_PATH = BASE_DATA_DIR.joinpath('metadata.csv')
TRAIN_IMG_PATH = BASE_DATA_DIR.joinpath('train', 'train-org-img')
TRAIN_LABEL_PATH = BASE_DATA_DIR.joinpath('train', 'train-label-img')
VAL_IMG_PATH = BASE_DATA_DIR.joinpath('val', 'val-org-img')
VAL_LABEL_PATH = BASE_DATA_DIR.joinpath('val', 'val-label-img')
TEST_IMG_PATH = BASE_DATA_DIR.joinpath('test', 'test-org-img')
TEST_LABEL_PATH = BASE_DATA_DIR.joinpath('test', 'test-label-img')
PATCH_DATA_CSV_FILE_FMT = 'patch_{:d}_data.csv'


def _get_patch_data_csv_path(cell_size):
    return BASE_DATA_DIR.joinpath('patch_{:d}_data.csv'.format(cell_size))


def get_dataset_list():
    return [
        FloodNetDatasetResNet, FloodNetDatasetUNet224, FloodNetDatasetUNet448,
        FloodNetDataset8Small, FloodNetDataset8Medium, FloodNetDataset8Large,
        FloodNetDataset16Small, FloodNetDataset16Medium, FloodNetDataset16Large,
        FloodNetDataset32Small, FloodNetDataset32Medium, FloodNetDataset32Large,
        FloodNetDatasetMultiResSingleOut, FloodNetDatasetMultiResMultiOut
    ]


def setup(patch_details):
    metadata_df = _load_metadata_df()

    # df = pd.DataFrame(columns=['image_id', 'split', 'width', 'height', 'ratio'])
    # for i in tqdm(range(len(metadata_df)), desc='Calculating patch targets', leave=False):
    #     # Load mask data
    #     image_id = metadata_df['image_id'][i]
    #     split = metadata_df['split'][i]
    #     mask_path = metadata_df['mask_path'][i]
    #     mask_img = Image.open(mask_path)
    #     width, height = mask_img.size
    #     df.loc[len(df)] = [image_id, split, width, height, width/height]
    # for split in ['train', 'val', 'test']:
    #     split_df = df[df['split'] == split]
    #     print(split)
    #     print(np.unique(split_df['width'], return_counts=True))
    #     print(np.unique(split_df['height'], return_counts=True))
    #     print(np.unique(split_df['ratio'], return_counts=True))
    #     print(split_df.loc[split_df['width'] == 4592, 'image_id'])
    # exit(0)
    _setup_patch_csv(metadata_df, patch_details)
    # calculate_dataset_normalisation(metadata_df, "img_path")


def _load_metadata_df():
    # Create metadata df if it doesn't exist already (not provided with the original dataset)
    print('Loading metadata')
    if not os.path.exists(METADATA_PATH):
        print(" Creating metadata file as it doesn't exist")
        columns = ['image_id', 'split', 'img_path', 'mask_path'] + FloodNetDataset.clz_names
        metadata_df = pd.DataFrame(columns=columns)

        def _get_split_data(img_dir_path, label_dir_path, split):
            rows = []
            for img_file in tqdm(os.listdir(img_dir_path), desc='Parsing metadata for {:s} split'.format(split)):
                # Load path data
                img_path = img_dir_path.joinpath(img_file)
                img_id = int(img_path.stem)
                label_file = '{:d}_lab.png'.format(img_id)
                label_path = label_dir_path.joinpath(label_file)
                assert os.path.exists(img_path) and os.path.exists(label_path)
                row = [img_id, split, img_path, label_path]

                # Compute coverage
                label_img = Image.open(str(label_path))
                label_arr = np.array(label_img)
                targets = np.zeros(10)
                found_clzs, clz_counts = np.unique(label_arr, return_counts=True)
                for idx, clz in enumerate(found_clzs):
                    targets[clz] = clz_counts[idx] / np.sum(clz_counts)
                if abs(sum(targets) - 1) > 1e-6:
                    raise AssertionError("Label targets don't sum to one (within reasonable error): {:} {:} {:}"
                                         .format(img_id, targets, abs(sum(targets) - 1)))
                row += targets.tolist()
                rows.append(row)

            df = pd.DataFrame(data=rows, columns=columns)
            df = df.sort_values('image_id')
            return df

        train_df = _get_split_data(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, 'train')
        metadata_df = pd.concat([metadata_df, train_df], ignore_index=True)
        val_df = _get_split_data(VAL_IMG_PATH, VAL_LABEL_PATH, 'val')
        metadata_df = pd.concat([metadata_df, val_df], ignore_index=True)
        test_df = _get_split_data(TEST_IMG_PATH, TEST_LABEL_PATH, 'test')
        metadata_df = pd.concat([metadata_df, test_df], ignore_index=True)
        metadata_df.to_csv(METADATA_PATH, index=False)
    else:
        metadata_df = pd.read_csv(METADATA_PATH)
    print(' Found {:d} images ({:d}/{:d}/{:d})'.format(len(metadata_df),
                                                       len(metadata_df[metadata_df['split'] == 'train']),
                                                       len(metadata_df[metadata_df['split'] == 'val']),
                                                       len(metadata_df[metadata_df['split'] == 'test'])))
    return metadata_df


def _setup_patch_csv(metadata_df, patch_details):
    """
    Extract cell labels from the training images.
    :param metadata_df: Dataframe of image metadata.
    """
    print('Setting up patch csv')
    print(' Cell size: {:d} x {:d}'.format(patch_details.cell_width, patch_details.cell_height))
    print(' Patch size: {:d}'.format(patch_details.patch_size))
    print(' Grid: {:d} x {:d}'.format(patch_details.grid_n_rows, patch_details.grid_n_cols))
    print(' {:d} patches per image'.format(patch_details.num_patches))
    print(' {:d} x {:d} effective cell resolution'.format(patch_details.effective_cell_resolution_width,
                                                          patch_details.effective_cell_resolution_height))
    print(' {:d} x {:d} effective patch resolution'.format(patch_details.effective_patch_resolution_width,
                                                           patch_details.effective_patch_resolution_height))
    print(' {:.2f}% scale'.format(patch_details.scale * 100))

    all_patch_data = []
    # Loop over all the images
    for i in tqdm(range(len(metadata_df)), desc='Calculating patch targets', leave=False):
        # Load mask data
        image_id = metadata_df['image_id'][i]
        mask_path = metadata_df['mask_path'][i]
        mask_img = Image.open(mask_path)
        mask_img = mask_img.resize((patch_details.effective_cell_resolution_width,
                                    patch_details.effective_cell_resolution_height),
                                   resample=Image.Resampling.NEAREST)
        mask_img_arr = np.array(mask_img)

        # Iterate through each cell in the grid
        for i_row in range(patch_details.grid_n_rows):
            for i_col in range(patch_details.grid_n_cols):
                # Extract patch from original image
                p_row = i_row * patch_details.cell_width
                p_col = i_col * patch_details.cell_height
                # Extract mask patch from original mask
                patch_mask_arr = mask_img_arr[
                                    p_row:p_row+patch_details.cell_width,
                                    p_col:p_col+patch_details.cell_height
                                 ]

                # Get clz coverage in this image
                patch_targets = np.zeros(10)
                found_clzs, clz_counts = np.unique(patch_mask_arr, return_counts=True)
                for idx, clz in enumerate(found_clzs):
                    patch_targets[clz] = clz_counts[idx] / np.sum(clz_counts)
                if abs(sum(patch_targets) - 1) > 1e-6:
                    raise AssertionError("Patch targets don't sum to one (within reasonable error): {:} {:} {:} {:} {:}"
                                         .format(image_id, i_row, i_col, patch_targets, abs(sum(patch_targets) - 1)))

                patch_data = [image_id, i_row, i_col] + patch_targets.tolist()
                all_patch_data.append(patch_data)

    # Save the patch dataframe
    df_cols = ['image_id', 'i_x', 'i_y'] + [FloodNetDataset.label_to_name(i) for i in range(10)]
    patches_df = pd.DataFrame(data=all_patch_data, columns=df_cols)
    patches_df.to_csv(_get_patch_data_csv_path(patch_details.cell_width), index=False)


class FloodNetDataset(MilDataset, ABC):

    # Handled by models?
    d_in = None
    n_expected_dims = 4  # i x c x h x w
    n_classes = 10
    metric_clz = RegressionMetric
    dataset_mean = (0.4111, 0.4483, 0.3415)
    dataset_std = (0.1260, 0.1185, 0.1177)
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
    # Clz names in correct order (i.e., clz 0 = background
    clz_names = ['Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', 'Road-non-flooded',
                 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
    cmap = mpl.colors.ListedColormap([plt.cm.tab10.colors[i] for i in [7, 3, 1, 4, 6, 9, 2, 5, 0, 8]])

    def __init__(self, bags, targets, instance_targets, bags_metadata):
        super().__init__(bags, targets, instance_targets, bags_metadata)
        self.transform = self.basic_transform

    @classmethod
    @property
    @abstractmethod
    def model_type(cls) -> str:
        pass

    @classmethod
    @property
    @abstractmethod
    def patch_details(cls) -> PatchDetails:
        pass

    @classmethod
    def load_bags(cls, split):
        metadata_df = _load_metadata_df()
        split_df = metadata_df[metadata_df['split'] == split]
        bags = split_df['img_path'].to_numpy()
        instance_targets = cls._parse_instance_targets(split_df)
        targets = split_df[cls.clz_names].to_numpy()
        mask_paths = split_df['mask_path'].to_numpy()

        # Set up bag metadata with bag id, grid size x, and grid size y
        bags_metadata = []
        for idx, bag_id in enumerate(split_df['image_id'].tolist()):
            bag_metadata = {
                'id': bag_id,
                'grid_n_rows': cls.patch_details.grid_n_rows,
                'grid_n_cols': cls.patch_details.grid_n_cols,
                'mask_path': mask_paths[idx]
            }
            bags_metadata.append(bag_metadata)
        bags_metadata = np.asarray(bags_metadata)

        return bags, targets, instance_targets, bags_metadata

    @classmethod
    def _parse_instance_targets(cls, split_df, cell_width=None):
        if cell_width is None:
            cell_width = cls.patch_details.cell_width
        patches_df = pd.read_csv(_get_patch_data_csv_path(cell_width))
        # coverage_df = cls.load_per_class_coverage()
        instance_targets = []
        for image_id in split_df['image_id']:
            image_patch_data = patches_df.loc[patches_df['image_id'] == image_id]
            bag_instance_targets = image_patch_data[cls.clz_names].to_numpy()
            instance_targets.append(torch.as_tensor(bag_instance_targets))
        return torch.stack(instance_targets)

    @classmethod
    def label_to_name(cls, label: int):
        if 0 <= label < len(cls.clz_names):
            return cls.clz_names[label]
        raise ValueError('No class name found for label {:d}'.format(label))

    @classmethod
    def create_complete_dataset(cls):
        pass

    @classmethod
    def get_target_mask(cls, instance_targets, clz):
        pass

    @classmethod
    def create_datasets(cls, random_state=12):
        # Dataset split is the same every time (as defined in the original dataset)
        #  Therefore just yield the same datasets each time, achieved with an infinite loop
        #  This assumes the caller correctly stops executing, which is default trainer does
        #   (both for single and repeat trainings)
        while True:
            train_bags, train_targets, train_its, train_md = cls.load_bags('train')
            train_dataset = cls(train_bags, train_targets, train_its, train_md)
            val_bags, val_targets, val_its, val_md = cls.load_bags('val')
            val_dataset = cls(val_bags, val_targets, val_its, val_md)
            test_bags, test_targets, test_its, test_md = cls.load_bags('test')
            test_dataset = cls(test_bags, test_targets, test_its, test_md)
            yield train_dataset, val_dataset, test_dataset

    @staticmethod
    def mask_img_to_clz_tensor(mask_img):
        mask_clz_tensor = torch.as_tensor(np.array(mask_img))
        return mask_clz_tensor

    @classmethod
    def clz_idx_to_rgb(cls, clz_idx):
        rgba = cls.cmap(clz_idx/cls.n_classes)
        return rgba

    def get_img(self, bag_idx):
        sat_path = self.bags[bag_idx]
        img = Image.open(sat_path)
        return img

    def get_mask_img(self, bag_idx):
        mask_img = Image.open(self.bags_metadata[bag_idx]['mask_path'])
        return mask_img

    def __getitem__(self, bag_idx):
        # Load original satellite and mask images
        sat_path = self.bags[bag_idx]

        # Resize to effective patch resolution before extracting
        img = Image.open(sat_path)
        img = img.resize((self.patch_details.effective_patch_resolution_width,
                          self.patch_details.effective_patch_resolution_height))
        img_arr = np.array(img)

        # Iterate through each cell in the grid
        instances = []
        for i_row in range(self.patch_details.grid_n_rows):
            for i_col in range(self.patch_details.grid_n_cols):
                # Extract patch from original image
                p_row = i_row * self.patch_details.patch_size
                p_col = i_col * self.patch_details.patch_size
                instance = img_arr[p_row:p_row+self.patch_details.patch_size,
                                   p_col:p_col+self.patch_details.patch_size,
                                   :]
                if self.transform is not None:
                    instance = self.transform(instance)
                instances.append(instance)

        # Get bag and metadata
        bag = torch.stack(instances)
        metadata = self.bags_metadata[bag_idx]

        # Reshape instance targets to a grid
        bag_instance_targets = self._get_instance_targets(bag_idx, metadata)

        # Return required data as a dict
        data_dict = {
            'bag': bag,
            'target': self.targets[bag_idx],
            'instance_targets': bag_instance_targets,
            'bag_metadata': metadata,
        }
        return data_dict

    def _get_instance_targets(self, bag_idx, metadata):
        # Reshape instance targets to a grid
        bag_instance_targets = self.instance_targets[bag_idx]
        bag_instance_targets = bag_instance_targets.swapaxes(0, 1)
        bag_instance_targets = bag_instance_targets.reshape(-1, metadata['grid_n_rows'], metadata['grid_n_cols'])
        return bag_instance_targets


class FloodNetDatasetResNet(FloodNetDataset):
    model_type = "resnet"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(1, 1, 224, 4000, 3000)


class FloodNetDatasetUNet224(FloodNetDataset):
    model_type = "unet224"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(1, 1, 224, 4000, 3000)


class FloodNetDatasetUNet448(FloodNetDataset):
    model_type = "unet448"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(1, 1, 448, 4000, 3000)


class FloodNetDataset8Small(FloodNetDataset):
    model_type = "8_small"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(6, 8, 28, 4000, 3000)


class FloodNetDataset16Small(FloodNetDataset):
    model_type = "16_small"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(12, 16, 28, 4000, 3000)


class FloodNetDataset32Small(FloodNetDataset):
    model_type = "32_small"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(24, 32, 28, 4000, 3000)


class FloodNetDataset8Medium(FloodNetDataset):
    model_type = "8_medium"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(6, 8, 56, 4000, 3000)


class FloodNetDataset16Medium(FloodNetDataset):
    model_type = "16_medium"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(12, 16, 56, 4000, 3000)


class FloodNetDataset32Medium(FloodNetDataset):
    model_type = "32_medium"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(24, 32, 56, 4000, 3000)


class FloodNetDataset8Large(FloodNetDataset):
    model_type = "8_large"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(6, 8, 102, 4000, 3000)


class FloodNetDataset16Large(FloodNetDataset):
    model_type = "16_large"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(12, 16, 102, 4000, 3000)


class FloodNetDataset32Large(FloodNetDataset):
    model_type = "32_large"
    name = 'floodnet_' + model_type
    patch_details = PatchDetails(24, 32, 102, 4000, 3000)


class FloodNetDatasetMultiResSingleOut(FloodNetDataset):
    model_type = "multi_res_single_out"
    name = "floodnet_" + model_type
    patch_details = PatchDetails(6, 8, 500, 4000, 3000)

    @classmethod
    @overrides
    def load_bags(cls, split):
        bags, targets, _, bags_metadata = super().load_bags(split)
        metadata_df = _load_metadata_df()
        split_df = metadata_df[metadata_df['split'] == split]
        # Replace default instance targets with smallest patch size (scale=m)
        instance_targets = cls._parse_instance_targets(split_df, cell_width=125)
        # Replace default metadata with correct spec for small patch size (scale=m)
        for m in bags_metadata:
            m['grid_n_rows'] = 24
            m['grid_n_cols'] = 32
        return bags, targets, instance_targets, bags_metadata


class FloodNetDatasetMultiResMultiOut(FloodNetDataset):
    model_type = "multi_res_multi_out"
    name = "floodnet_" + model_type
    patch_details = PatchDetails(6, 8, 500, 4000, 3000)

    @classmethod
    @overrides
    def load_bags(cls, split):
        bags, targets, s0_instance_targets, bags_metadata = super().load_bags(split)
        metadata_df = _load_metadata_df()
        split_df = metadata_df[metadata_df['split'] == split]
        # Replace default instance targets with smallest patch size (scale=m)
        s1_instance_targets = cls._parse_instance_targets(split_df, cell_width=250)
        s2_instance_targets = cls._parse_instance_targets(split_df, cell_width=125)
        multires_targets = []
        for i in range(len(s0_instance_targets)):
            multires_targets.append([
                s0_instance_targets[i],
                s1_instance_targets[i],
                s2_instance_targets[i]
            ])
        # Replace default metadata with correct spec for small patch size (scale=m)
        for m in bags_metadata:
            del m['grid_n_rows']
            del m['grid_n_cols']
            m['s0_grid_n_rows'] = 6
            m['s0_grid_n_cols'] = 8
            m['s1_grid_n_rows'] = 12
            m['s1_grid_n_cols'] = 16
            m['s2_grid_n_rows'] = 24
            m['s2_grid_n_cols'] = 32
        return bags, targets, multires_targets, bags_metadata

    @overrides
    def _get_instance_targets(self, bag_idx, metadata):
        # Reshape instance targets to a grid for each scale
        bag_instance_targets = self.instance_targets[bag_idx]
        reshaped_instance_targets = []
        for i in range(3):
            s_bag_instance_targets = bag_instance_targets[i]
            # Originally (instance, class). Transform to (class, grid_row, grid_col)
            s_bag_instance_targets = s_bag_instance_targets.swapaxes(0, 1)
            s_bag_instance_targets = s_bag_instance_targets.reshape(-1,
                                                                    metadata['s{:d}_grid_n_rows'.format(i)],
                                                                    metadata['s{:d}_grid_n_cols'.format(i)])
            reshaped_instance_targets.append(s_bag_instance_targets)
        return reshaped_instance_targets


if __name__ == "__main__":
    setup(FloodNetDataset8Small.patch_details)
    setup(FloodNetDataset16Small.patch_details)
    setup(FloodNetDataset32Small.patch_details)
    setup(FloodNetDatasetResNet.patch_details)