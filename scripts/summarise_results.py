import csv

import latextable
import numpy as np
from texttable import Texttable


def parse_raw_results(file, model_names, n_expected_blocks, include_patch_size_in_names, n_dec):
    with open(file, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        # Parse three blocks: rmse/mae, grid_seg, and high_res_seg
        #  Aggregate the rows in each block, then split the rows in that block and reset.
        block = []
        block_num = 0
        for row in reader:
            block.append(row)
            # Each block should contain n_models * 2 + 4 lines
            #  Each model row has two lines (data row and table border)
            #  Four additional lines for the title (one line) and table header (three lines)
            if len(block) == n_expected_blocks * 2 + 4:
                if block_num == 0:
                    scene_rmse = parse_split(block, -3)
                    scene_mae = parse_split(block, -2)
                elif block_num == 1:
                    grid_seg = parse_split(block, -2)
                elif block_num == 2:
                    high_res_seg = parse_split(block, -2)
                    break
                block_num += 1
                block = []
                next(reader)

    rows = []
    means = []
    for model in model_names:
        row = [format_model_type(model, include_patch_size=include_patch_size_in_names), scene_rmse[model],
               scene_mae[model], grid_seg[model], high_res_seg[model]]
        row_means = [float(r.split(' +- ')[0]) for r in row[1:]]
        means.append(row_means)
        # n_dec: Number of decimal points
        fmt_str = '{:.' + str(n_dec) + 'f} $\pm$ {:.' + str(n_dec) + 'f}'
        row[1:] = [fmt_str.format(*[float(s) for s in r.split(' +- ')]) for r in row[1:]]
        row = [r if 'nan' not in r else 'N/A' for r in row]
        rows.append(row)

    return rows, means


def parse_results(dataset, n_dec, reduced=False):
    # Aggregate lists of data parsed from all files
    rows = [['Configuration', 'Scene RMSE', 'Scene MAE', 'Patch mIoU', 'Pixel mIoU']]
    means = []

    if dataset == 'dgr':
        single_res_file = "results/DeepGlobe/single_res_raw_results_dgr.txt"
        multi_res_single_out_file = "results/DeepGlobe/multi_res_single_out_raw_results_dgr.txt"
        multi_res_multi_out_file = "results/DeepGlobe/multi_res_multi_out_raw_results_dgr.txt"
    elif dataset == 'floodnet':
        single_res_file = "results/FloodNet/single_res_raw_results_floodnet.txt"
        multi_res_single_out_file = "results/FloodNet/multi_res_single_out_raw_results_floodnet.txt"
        multi_res_multi_out_file = "results/FloodNet/multi_res_multi_out_raw_results_floodnet.txt"
    else:
        raise NotImplementedError

    # Parse single res results
    if reduced:
        single_res_model_names = ['resnet', 'unet224', 'unet448', '8_large', '16_large', '32_large']
    else:
        single_res_model_names = ['resnet', 'unet224', 'unet448', '8_small', '8_medium', '8_large', '16_small',
                                  '16_medium', '16_large', '32_small', '32_medium', '32_large']
    single_res_out_rows, single_res_out_means = parse_raw_results(single_res_file, single_res_model_names, 12,
                                                                  include_patch_size_in_names=not reduced, n_dec=n_dec)
    rows += single_res_out_rows
    means += single_res_out_means

    # Parse multi res single out results
    multi_res_single_out_modes = ['multi_res_single_out']
    multi_res_single_out_rows, multi_res_single_out_means = parse_raw_results(multi_res_single_out_file,
                                                                              multi_res_single_out_modes, 1,
                                                                              include_patch_size_in_names=not reduced,
                                                                              n_dec=n_dec)
    rows += multi_res_single_out_rows
    means += multi_res_single_out_means

    # Parse multi res multi out results
    multi_res_multi_out_models = ['s=0', 's=1', 's=2', 's=m']
    multi_res_multi_out_rows, multi_res_multi_out_means = parse_raw_results(multi_res_multi_out_file,
                                                                            multi_res_multi_out_models, 4,
                                                                            include_patch_size_in_names=not reduced,
                                                                            n_dec=n_dec)
    rows += multi_res_multi_out_rows
    means += multi_res_multi_out_rows

    n_models = len(single_res_model_names) + len(multi_res_single_out_modes) + len(multi_res_multi_out_models)

    for c in range(4):
        col_idx = c + 1
        cell_values = []
        for r in range(n_models):
            row_idx = r + 1
            cell_value = rows[row_idx][col_idx]
            if cell_value == 'N/A':
                cell_values.append(np.nan)
            else:
                cell_value = float(cell_value[:6])
                cell_values.append(cell_value)

        best_val = np.nanmin(cell_values) if col_idx < 3 else np.nanmax(cell_values)
        best_idxs = [r + 1 for r in range(n_models) if cell_values[r] == best_val]
        for best_idx in best_idxs:
            rows[best_idx][col_idx] = '\\textbf{' + rows[best_idx][col_idx] + '}'

    return rows


def run(dataset, reduced=False):
    # Parse results with 4 decimal places and create texttable table
    texttable_rows = parse_results(dataset, 4, reduced=reduced)
    tt_table = Texttable()
    tt_table.set_cols_dtype(['t'] * 5)
    tt_table.set_cols_align(['l'] * 5)
    tt_table.add_rows(texttable_rows)
    tt_table.set_max_width(0)

    # Parse results with 3 decimal places and create latextable table
    latex_rows = parse_results(dataset, 3, reduced=reduced)
    lt_table = Texttable()
    lt_table.set_cols_dtype(['t'] * 5)
    lt_table.set_cols_align(['l'] * 5)
    lt_table.add_rows(latex_rows)
    lt_table.set_max_width(0)

    return tt_table, lt_table


def parse_split(split, idx):
    data = split[4::2]
    model_names = [row[1].strip() for row in data]
    values = [row[idx].strip() for row in data]
    values_dict = dict(zip(model_names, values))
    return values_dict


def format_model_type(model_type, include_patch_size):
    if model_type == 'resnet':
        return 'ResNet18'
    elif 'unet' in model_type:
        return 'U-Net {:s}'.format(model_type[-3:])
    elif model_type == 'multi_res_single_out':
        return 'S2P MRSO s=m'
    elif 's=' in model_type:
        return 'S2P MRMO ' + model_type

    grid_size, patch_size = model_type.split('_')
    if grid_size == '8':
        s = 0
    elif grid_size == '16':
        s = 1
    elif grid_size == '32':
        s = 2
    else:
        raise NotImplementedError
    if include_patch_size:
        return 'S2P SR {:s} s={:d}'.format(patch_size.title(), s)
    return 'S2P SR s={:d}'.format(s)


if __name__ == "__main__":
    alias = {
        's=0': '$s=0$',
        's=1': '$s=1$',
        's=2': '$s=2$',
        's=m': '$s=m$',
    }

    # Summarise DeepGlobe results
    print('-- DeepGlobe --')
    complete_dgr_tt_table, complete_dgr_lt_table = run("dgr", reduced=False)
    reduced_dgr_tt_table, reduced_dgr_lt_table = run("dgr", reduced=True)
    # Text table output
    print('- Complete -')
    print(complete_dgr_tt_table.draw())
    print('\n- Reduced -')
    print(reduced_dgr_tt_table.draw())
    # Latex output
    print('- Complete -')
    print(latextable.draw_latex(complete_dgr_lt_table, use_booktabs=True, alias=alias))
    print('\n- Reduced -')
    print(latextable.draw_latex(reduced_dgr_lt_table, use_booktabs=True, alias=alias))

    # Summarise FloodNet results
    print('\n-- FloodNet --')
    complete_floodnet_tt_table, complete_floodnet_lt_table = run("floodnet", reduced=False)
    reduced_floodnet_tt_table, reduced_floodnet_lt_table = run("floodnet", reduced=True)
    # Text table output
    print('- Complete -')
    print(complete_floodnet_tt_table.draw())
    print('\n- Reduced -')
    print(reduced_floodnet_tt_table.draw())
    # Latex output
    print('- Complete -')
    print(latextable.draw_latex(complete_floodnet_lt_table, use_booktabs=True, alias=alias))
    print('\n- Reduced -')
    print(latextable.draw_latex(reduced_floodnet_lt_table, use_booktabs=True, alias=alias))
