import numpy as np
from PIL import Image
from tqdm import tqdm


class PatchDetails:

    def __init__(self, grid_size_x, grid_size_y, patch_size, orig_img_size_x, orig_img_size_y):
        # Size of grid to apply over the original image
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        # Size to reduce each cell of the grid to (assume square patches)
        self.patch_size = patch_size
        # Size of original image
        self.orig_img_size_x = orig_img_size_x
        self.orig_img_size_y = orig_img_size_y
        # Size of each cell (according to original image size and grid size)
        self.cell_size = self._calculate_cell_size()
        # Total number of patches that will be extracted from each image (size of grid squared)
        self.num_patches = self._calculate_num_patches()
        # Effective resolution after extracting cells
        self.effective_cell_resolution_x, self.effective_cell_resolution_y = self._calculate_effective_cell_resolution()
        # Effective resolution after extracting and resizing cells (to get patches)
        self.effective_patch_resolution_x, self.effective_patch_resolution_y = self._calculate_effective_patch_resolution()
        # Scale of effective resolution compared to original resolution
        self.scale = self._calculate_scale()
        # Check values are okay
        self._check_valid()

    def _check_valid(self):
        if self.grid_size_x * self.patch_size > self.orig_img_size_x or \
                self.grid_size_y * self.patch_size > self.orig_img_size_y:
            raise ValueError(
                'Cannot use grid_size {:d} x {:d} and patch_size {:d} as effective patch resolution is larger than the'
                ' original image ({:d} x {:d} > {:d} x {:d}_'
                .format(self.grid_size_x, self.grid_size_y, self.patch_size,
                        self.effective_patch_resolution_x, self.effective_patch_resolution_y,
                        self.orig_img_size_x, self.orig_img_size_y)
            )

    def _calculate_cell_size(self):
        # Calculate largest possible cell size (whole number) at scale of the original image
        cell_size_x = self.orig_img_size_x // self.grid_size_x
        cell_size_y = self.orig_img_size_y // self.grid_size_y
        if cell_size_x != cell_size_y:
            raise ValueError('Invalid configuration: cell size must be the same in x and y (currently {:d} and {:d}'
                             .format(cell_size_x, cell_size_y))
        return cell_size_x

    def _calculate_num_patches(self):
        return self.grid_size_x * self.grid_size_y

    def _calculate_effective_cell_resolution(self):
        return self.grid_size_x * self.cell_size, self.grid_size_y * self.cell_size

    def _calculate_effective_patch_resolution(self):
        return self.grid_size_x * self.patch_size, self.grid_size_y * self.patch_size

    def _calculate_scale(self):
        n_px_scaled = self.effective_patch_resolution_x * self.effective_patch_resolution_y
        n_px_orig = self.orig_img_size_x * self.orig_img_size_y
        return n_px_scaled / n_px_orig


class PatchDetailsSquare(PatchDetails):

    def __init__(self, grid_size, patch_size, orig_img_size):
        super().__init__(grid_size, grid_size, patch_size, orig_img_size, orig_img_size)

    @property
    def grid_size(self):
        return self.grid_size_x

    @property
    def orig_img_size(self):
        return self.orig_img_size_x


def get_model_type_list(dataset_list):
    return [dataset_clz.model_type for dataset_clz in dataset_list]


def get_dataset_clz(model_type, dataset_list):
    for dataset_clz in dataset_list:
        if dataset_clz.model_type == model_type:
            return dataset_clz


def calculate_dataset_normalisation(metadata_df, img_path_col='sat_image_path'):
    print('Calculating dataset normalisation')
    avgs = []
    for i in tqdm(range(len(metadata_df)), desc='Calculating dataset normalisation', leave=False):
        sat_path = metadata_df[img_path_col][i]
        sat_img = Image.open(sat_path)
        sat_img_arr = np.array(sat_img) / 255
        avg = np.mean(sat_img_arr, axis=(0, 1))
        avgs.append(avg)
    arrs = np.stack(avgs)
    arrs_mean = np.mean(arrs, axis=0)
    arrs_std = np.std(arrs, axis=0)
    print(' Mean:', arrs_mean)
    print('  Std:', arrs_std)
