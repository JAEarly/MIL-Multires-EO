import argparse

import latextable
import numpy as np
import wandb
from texttable import Texttable

from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device, load_model
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz
from dgr_luc_models import get_model_clz
from dgr_luc_multires_trainer import MultiResTrainer
import torch
from evaluate_luc_models import evaluate_iou_grid, evaluate_iou_segmentation

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-resolution model evaluation script.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.n_repeats


def run_evaluation():
    wandb.init()

    n_repeats = parse_args()
    model_type = "multi_res"
    n_scales = 4  # (0, 1, 2, m)

    # print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for model: {:}'.format(model_type))

    model_clz = get_model_clz(model_type)
    dataset_clz = get_dataset_clz(model_type)

    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_type)
    wandb.config.update(training_config, allow_val_change=True)

    trainer = create_trainer_from_clzs(device, model_clz, dataset_clz,
                                       dataloader_func=create_normal_dataloader, trainer_clz=MultiResTrainer)

    bag_results, grid_seg_results, orig_seg_results = evaluate(n_scales, n_repeats, trainer)

    scale_names = ['s=0', 's=1', 's=2', 's=m']
    print('\nBag Results')
    output_multi_res_bag_results(scale_names, bag_results, 4, sort=False)
    print('\nGrid Segmentation Results')
    output_iou_results(scale_names, grid_seg_results, sort=False)
    print('\nHigh-Res Segmentation Results')
    output_iou_results(scale_names, orig_seg_results, sort=False)


def evaluate(n_scales, n_repeats, trainer, random_state=5):
    bag_results = np.empty((n_repeats, 3), dtype=object)
    grid_seg_results = np.empty((n_repeats, 3, n_scales), dtype=object)
    orig_seg_results = np.empty((n_repeats, 3, n_scales), dtype=object)

    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, True, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(device, trainer.dataset_clz.name, trainer.model_clz, modifier=r)

        results_list = eval_complete(trainer, model, train_dataloader, val_dataloader, test_dataloader, verbose=False)

        train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res = results_list
        bag_results[r, :] = [train_bag_res[0], val_bag_res[0], test_bag_res[0]]
        grid_seg_results[r, :] = [train_inst_res[0], val_inst_res[0], test_inst_res[0]]
        orig_seg_results[r, :] = [train_inst_res[1], val_inst_res[1], test_inst_res[1]]

        r += 1
        if r == n_repeats:
            break

    return bag_results, grid_seg_results, orig_seg_results


def eval_complete(trainer, model, train_dataloader, val_dataloader, test_dataloader, verbose=False):
    train_bag_res, train_inst_res = eval_model(trainer, model, train_dataloader, verbose=verbose)
    val_bag_res, val_inst_res = eval_model(trainer, model, val_dataloader, verbose=verbose)
    test_bag_res, test_inst_res = eval_model(trainer, model, test_dataloader, verbose=verbose)
    return train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res


def eval_model(trainer, model, dataloader, verbose=False):
    # Iterate through data loader and gather preds and targets
    model_outs = trainer.get_model_outputs_for_dataset(model, dataloader, lim=None)
    all_preds, all_targets, all_instance_preds, all_instance_targets, all_mask_paths = model_outs
    labels = list(range(model.n_classes))

    # Calculate bag results
    bag_results = [trainer.metric_clz.calculate_metric(all_preds, all_targets, labels)]
    if verbose:
        for bag_result in bag_results:
            bag_result.out()

    # Calculate instance results
    all_grid_results = []
    all_seg_results = []
    for scale_idx in range(4):
        scale_instance_preds = torch.stack([p[scale_idx] for p in all_instance_preds])
        if scale_idx == 0:
            scale_instance_targets = torch.stack([t.squeeze()[:64, :] for t in all_instance_targets])
        elif scale_idx == 1:
            scale_instance_targets = torch.stack([t.squeeze()[64:64+256, :] for t in all_instance_targets])
        else:
            scale_instance_targets = torch.stack([t.squeeze()[256+64:, :] for t in all_instance_targets])

        # Wrangle targets to grid shape
        #  Swap class and instance axes
        #  Reshape to match the image grid
        grid_targets = scale_instance_targets\
            .swapaxes(1, 2)\
            .reshape(-1, len(labels), scale_instance_preds.shape[2], scale_instance_preds.shape[3])
        grid_results = evaluate_iou_grid(scale_instance_preds, grid_targets, labels)
        seg_results = evaluate_iou_segmentation(dataloader.dataset, scale_instance_preds, labels, all_mask_paths)

        all_grid_results.append(grid_results)
        all_seg_results.append(seg_results)
        if verbose:
            grid_results.out()
            seg_results.out()

    instance_results = [all_grid_results, all_seg_results]

    return bag_results, instance_results


def output_multi_res_bag_results(model_names, results_arr, n_scales, sort=True, latex=False):
    n_repeats, _ = results_arr.shape
    results = np.empty((n_scales, 6), dtype=object)
    mean_test_mae_losses = []

    for scale_idx in range(n_scales):
        scale_results = np.empty((n_repeats, 6), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = results_arr[repeat_idx]
            scale_results[repeat_idx, 0] = train_results.rmse_losses[scale_idx]
            scale_results[repeat_idx, 1] = train_results.mae_losses[scale_idx]
            scale_results[repeat_idx, 2] = val_results.rmse_losses[scale_idx]
            scale_results[repeat_idx, 3] = val_results.mae_losses[scale_idx]
            scale_results[repeat_idx, 4] = test_results.rmse_losses[scale_idx]
            scale_results[repeat_idx, 5] = test_results.mae_losses[scale_idx]
        mean = np.mean(scale_results, axis=0)
        sem = np.std(scale_results, axis=0) / np.sqrt(len(scale_results))
        mean_test_mae_losses.append(mean[5])
        for metric_idx in range(6):
            results[scale_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])

    model_order = np.argsort(mean_test_mae_losses) if sort else list(range(len(model_names)))
    rows = [['Model Name', 'Train RMSE', 'Train MAE', 'Val RMSE', 'Val MAE', 'Test RMSE', 'Test MAE']]
    for model_idx in model_order:
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    if latex:
        print(latextable.draw_latex(table))


def output_iou_results(model_names, results_arr, sort=True, latex=False):
    n_repeats, _, n_scales = results_arr.shape
    results = np.empty((n_scales, 3), dtype=object)
    mean_test_ious = []
    for scale_idx in range(n_scales):
        model_results = results_arr[:, :, scale_idx]
        expanded_model_results = np.empty((n_repeats, 3), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.mean_iou,
                                                     val_results.mean_iou,
                                                     test_results.mean_iou]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        mean_test_ious.append(mean[2])
        for metric_idx in range(3):
            results[scale_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    model_order = np.argsort(mean_test_ious) if sort else list(range(len(model_names)))
    rows = [['Model Name', 'Train IoU', 'Val IoU', 'Test IoU']]
    for model_idx in model_order:
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 4)
    table.set_cols_align(['c'] * 4)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    if latex:
        print(latextable.draw_latex(table))


if __name__ == "__main__":
    run_evaluation()
