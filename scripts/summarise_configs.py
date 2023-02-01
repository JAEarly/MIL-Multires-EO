import wandb

from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from deepglobe import dgr_luc_dataset, dgr_luc_models
from floodnet import floodnet_dataset, floodnet_models
from texttable import Texttable

import time

from latextable import draw_latex


def summarise_configs(dataset_list, n_param_func):
    wandb.init()

    rows = [['Configuration', 'Grid Size', 'Cell Size', 'Patch Size', 'Eff. Resolution', 'Scale', '\\# Params']]
    for dataset in dataset_list:
        patch_details = dataset.patch_details
        model_type = dataset.model_type
        config_path = "config/model_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)
        row = [
            _format_model_type(model_type),
            "{:d} x {:d}".format(patch_details.grid_size_x, patch_details.grid_size_y),
            "{:d} x {:d} px".format(patch_details.cell_size, patch_details.cell_size),
            "{:d} x {:d} px".format(patch_details.patch_size, patch_details.patch_size),
            "{:d} x {:d} px".format(patch_details.effective_patch_resolution_x,
                                    patch_details.effective_patch_resolution_y),
            "{:.1f}\\%".format(patch_details.scale * 100),
            "{:s}".format(_format_n_params(n_param_func(model_type))),
        ]
        rows.append(row)

    table = Texttable()
    table.add_rows(rows)
    table.set_cols_align(['l'] * 7)
    table.set_max_width(0)
    print(table.draw())

    print('\n')
    print(draw_latex(table, use_booktabs=True))

    # For wandb
    time.sleep(3)


def _format_model_type(model_type):
    if model_type == 'resnet':
        return 'ResNet18'
    elif 'unet' in model_type:
        return 'UNet {:s}'.format(model_type[-3:])
    elif model_type == 'multi_res_single_out':
        return 'S2P Multi Res Single Out'
    elif model_type == 'multi_res_multi_out':
        return 'S2P Multi Res Multi Out'
    grid_size, patch_size = model_type.split('_')
    return 'S2P Single Res {:s} {:s}'.format(patch_size.title(), grid_size)


def _format_n_params(n_params):
    if n_params >= 1e6:
        return "{:.2f}M".format(n_params/1e6)
    else:
        return "{:.0f}K".format(n_params/1e3)


if __name__ == "__main__":
    # summarise_configs(dgr_luc_dataset.get_dataset_list(), dgr_luc_models.get_n_params)
    summarise_configs(floodnet_dataset.get_dataset_list(), floodnet_models.get_n_params)
