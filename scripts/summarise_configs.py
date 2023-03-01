import wandb

from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from deepglobe import dgr_luc_dataset, dgr_luc_models
from floodnet import floodnet_dataset, floodnet_models
from texttable import Texttable
from dataset import PatchDetails

import time

from latextable import draw_latex


def summarise_configs(dataset_name, dataset_list, n_param_func):
    wandb.init()

    rows = [['Configuration', 'Grid Size', 'Cell Size', 'Patch Size', 'Eff. Resolution', 'Scale', '\\# Params']]
    for dataset in dataset_list:
        patch_details = dataset.patch_details
        model_type = dataset.model_type
        config_path = "config/model_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)

        if ' 24' in dataset.model_type:
            continue
        if 'multi_res' in dataset.model_type:
            if dataset_name == 'DeepGlobe':
                patch_details = PatchDetails(32, 32, 76, 2448, 2448)
            elif dataset_name == 'FloodNet':
                patch_details = PatchDetails(24, 32, 102, 4000, 3000)
            else:
                raise NotImplementedError
        row = [
            _format_model_type(model_type),
            "{:d} x {:d}".format(patch_details.grid_n_cols, patch_details.grid_n_rows),
            "{:d} x {:d} px".format(patch_details.cell_width, patch_details.cell_height),
            "{:d} x {:d} px".format(patch_details.patch_size, patch_details.patch_size),
            "{:d} x {:d} px".format(patch_details.effective_patch_resolution_width,
                                    patch_details.effective_patch_resolution_height),
            "{:.1f}\\%".format(patch_details.scale * 100),
            "{:s}".format(_format_n_params(n_param_func(model_type))),
        ]
        rows.append(row)

    table = Texttable()
    table.add_rows(rows)
    table.set_cols_align(['l'] * 7)
    table.set_max_width(0)
    print(dataset_name)
    print(table.draw())

    print('\n')
    print(draw_latex(table, use_booktabs=True))

    # For wandb
    time.sleep(3)


def _format_model_type(model_type):
    if model_type == 'resnet':
        return 'ResNet18'
    elif 'unet' in model_type:
        return 'U-Net {:s}'.format(model_type[-3:])
    elif model_type == 'multi_res_single_out':
        return 'S2P MRSO s = m'
    elif model_type == 'multi_res_multi_out':
        return 'S2P MRMO s = m'
    grid_size, patch_size = model_type.split('_')
    return 'S2P SR {:s} {:s}'.format(patch_size.title(), grid_size)


def _format_n_params(n_params):
    if n_params >= 1e6:
        return "{:.2f}M".format(n_params/1e6)
    else:
        return "{:.0f}K".format(n_params/1e3)


if __name__ == "__main__":
    summarise_configs('DeepGlobe', dgr_luc_dataset.get_dataset_list(), dgr_luc_models.get_n_params)
    summarise_configs('FloodNet', floodnet_dataset.get_dataset_list(), floodnet_models.get_n_params)
