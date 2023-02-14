import numpy as np
from PIL import Image
from tqdm import tqdm


class PatchDetails:

    def __init__(self, grid_n_rows, grid_n_cols, patch_size, orig_img_width, orig_img_height):
        # Size of grid to apply over the original image
        self.grid_n_rows = grid_n_rows
        self.grid_n_cols = grid_n_cols
        # Size to reduce each cell of the grid to (assume square patches)
        self.patch_size = patch_size
        # Size of original image
        self.orig_img_width = orig_img_width
        self.orig_img_height = orig_img_height
        # Size of each cell (according to original image size and grid size)
        self.cell_width, self.cell_height = self._calculate_cell_size()
        # Total number of patches that will be extracted from each image (size of grid squared)
        self.num_patches = self._calculate_num_patches()
        # Effective resolution after extracting cells
        self.effective_cell_resolution_width, self.effective_cell_resolution_height = \
            self._calculate_effective_cell_resolution()
        # Effective resolution after extracting and resizing cells (to get patches)
        self.effective_patch_resolution_width, self.effective_patch_resolution_height = \
            self._calculate_effective_patch_resolution()
        # Scale of effective resolution compared to original resolution
        self.scale = self._calculate_scale()
        # Check values are okay
        self._check_valid()

    def _check_valid(self):
        if self.effective_patch_resolution_width > self.orig_img_width or \
                self.effective_patch_resolution_height > self.orig_img_height:
            raise ValueError(
                'Cannot use grid_size {:d} x {:d} and patch_size {:d} as effective patch resolution is larger than the'
                ' original image ({:d} x {:d} > {:d} x {:d}_'
                .format(self.grid_n_rows, self.grid_n_cols, self.patch_size,
                        self.effective_patch_resolution_width, self.effective_patch_resolution_height,
                        self.orig_img_width, self.orig_img_height)
            )

    def _calculate_cell_size(self):
        # Calculate largest possible cell size (whole number) at scale of the original image
        cell_width = self.orig_img_width // self.grid_n_cols
        cell_height = self.orig_img_height // self.grid_n_rows
        if cell_width != cell_height:
            print('WARNING: Cell has different width and height (currently {:d} and {:d}).'
                  'This may cause the aspect ratio to change.'.format(cell_width, cell_height))
        return cell_width, cell_height

    def _calculate_num_patches(self):
        return self.grid_n_rows * self.grid_n_cols

    def _calculate_effective_cell_resolution(self):
        return self.grid_n_cols * self.cell_width, self.grid_n_rows * self.cell_height

    def _calculate_effective_patch_resolution(self):
        return self.grid_n_cols * self.patch_size, self.grid_n_rows * self.patch_size

    def _calculate_scale(self):
        n_px_scaled = self.effective_patch_resolution_width * self.effective_patch_resolution_height
        n_px_orig = self.orig_img_width * self.orig_img_height
        return n_px_scaled / n_px_orig


class PatchDetailsSquare(PatchDetails):

    def __init__(self, grid_size, patch_size, orig_img_size):
        super().__init__(grid_size, grid_size, patch_size, orig_img_size, orig_img_size)

    @property
    def grid_size(self):
        return self.grid_n_rows

    @property
    def cell_size(self):
        return self.cell_width

    @property
    def orig_img_size(self):
        return self.orig_img_width

    @property
    def effective_patch_resolution(self):
        return self.effective_patch_resolution_width


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
