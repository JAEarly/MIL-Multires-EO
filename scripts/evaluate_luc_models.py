import argparse
import torch

import numpy as np
import wandb

from bonfire.train.metrics import eval_complete, output_results, JaccardIndexMetric
from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device, load_model
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, get_model_type_list
from dgr_luc_models import get_model_clz

device = get_device()
all_models = get_model_type_list()
model_type_choices = all_models + ['all']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('model_types', choices=model_type_choices, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.model_types, args.n_repeats


def run_evaluation():
    wandb.init()

    model_types, n_repeats = parse_args()

    if model_types == ['all']:
        model_types = all_models

    # print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for models: {:}'.format(model_types))

    bag_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    inst_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    for model_idx, model_type in enumerate(model_types):
        print('Evaluating {:s}'.format(model_type))

        model_clz = get_model_clz(model_type)
        dataset_clz = get_dataset_clz(model_type)

        config_path = "config/dgr_luc_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)

        if model_type == 'resnet':
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, dataloader_func=create_normal_dataloader)
        else:
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)
        model_bag_results, model_inst_results, model_seg_results = evaluate(model_type, n_repeats, trainer)
        bag_results[model_idx, :, :] = model_bag_results
        inst_results[model_idx, :, :] = model_inst_results
        seg_results[model_idx, :, :] = model_seg_results
    print('\nBag Results')
    output_results(model_types, bag_results, sort=False)
    print('\nInstance Results')
    output_results(model_types, inst_results, sort=False)
    print('\nSegmentation Results')
    output_results(model_types, seg_results, sort=False)


def evaluate(model_type, n_repeats, trainer, random_state=5):
    bag_results_arr = np.empty((n_repeats, 3), dtype=object)
    inst_results_arr = np.empty((n_repeats, 3), dtype=object)
    seg_results_arr = np.empty((n_repeats, 3), dtype=object)

    bag_metrics = (trainer.metric_clz,)
    instance_metrics = () if model_type == 'resnet' else (trainer.metric_clz, JaccardIndexMetric)

    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, True, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(device, trainer.dataset_clz.name, trainer.model_clz, modifier=r)

        results_list = eval_complete(model, train_dataloader, val_dataloader, test_dataloader,
                                     bag_metrics=bag_metrics, instance_metrics=instance_metrics, verbose=False)

        train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res = results_list
        bag_results_arr[r, :] = [train_bag_res[0], val_bag_res[0], test_bag_res[0]]
        if model_type != 'resnet':
            inst_results_arr[r, :] = [train_inst_res[0], val_inst_res[0], test_inst_res[0]]
            seg_results_arr[r, :] = [train_inst_res[1], val_inst_res[1], test_inst_res[1]]
        else:
            inst_results_arr[r, :] = trainer.metric_clz(torch.nan, torch.nan)
            seg_results_arr[r, :] = JaccardIndexMetric(torch.nan)

        r += 1
        if r == n_repeats:
            break

    return bag_results_arr, inst_results_arr, seg_results_arr


if __name__ == "__main__":
    run_evaluation()
