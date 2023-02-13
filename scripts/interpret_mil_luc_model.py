import argparse

import wandb

from bonfire.util import get_device
from bonfire.util import load_model_from_path, get_default_save_path
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dataset import get_dataset_clz, get_model_type_list
from deepglobe import dgr_luc_dataset
# from deepglobe.dgr_luc_interpretability import MilLucInterpretabilityStudy
from floodnet.floodnet_interpretability import FloodNetInterpretabilityStudy
from deepglobe.dgr_luc_models import get_model_clz as get_model_clz_dgr
from floodnet import floodnet_dataset
from floodnet.floodnet_models import get_model_clz as get_model_clz_floodnet

device = get_device()
dataset_choices = ["dgr", "floodnet"]


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC interpretability script.')
    parser.add_argument('dataset', choices=dataset_choices, help="Dataset to train on.")
    parser.add_argument('model_type', help="Type of model to interpret.")
    parser.add_argument('task', choices=['sample', 'specific'], help='The task to perform.')
    parser.add_argument('-s', '--show_outputs', action='store_true',
                        help="Whether or not to show the interpretability outputs (they're always saved).")
    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    if args.dataset == 'dgr':
        dataset_list = dgr_luc_dataset.get_dataset_list()
    elif args.dataset == 'floodnet':
        dataset_list = floodnet_dataset.get_dataset_list()
    else:
        raise ValueError('No dataset list for dataset {:s}'.format(args.dataset))

    model_type_choices = get_model_type_list(dataset_list)
    if args.model_type not in model_type_choices:
        raise ValueError('Model type {:s} not found in model types for dataset {:s}'
                         .format(args.model_type, args.dataset))

    if args.dataset == 'dgr':
        model_clz = get_model_clz_dgr(args.model_type)
    elif args.dataset == 'floodnet':
        model_clz = get_model_clz_floodnet(args.model_type)
    else:
        raise ValueError('No models for dataset {:s}'.format(args.dataset))
    dataset_clz = get_dataset_clz(args.model_type, dataset_list)

    if args.dataset == 'dgr':
        model_idx = dgr_get_best_model_idx(args.model_type)
    elif args.dataset == 'floodnet':
        # TODO best model idxs for floodnet
        model_idx = 0
    else:
        raise NotImplementedError

    model_path, _, _ = get_default_save_path(dataset_clz.name, model_clz.name, modifier=model_idx)

    # Parse wandb config and get training config for this model
    config_path = "config/model_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], args.model_type)
    wandb.init(
        config=training_config,
    )

    test_dataset = next(dataset_clz.create_datasets())[0]
    model = load_model_from_path(device, model_clz, model_path)
    study = FloodNetInterpretabilityStudy(device, test_dataset, model, args.show_outputs)

    if args.task == 'sample':
        study.sample_interpretations()
    elif args.task == 'specific':
        study.create_interpretation_from_id(941237)
    else:
        raise NotImplementedError('Task not implemented: {:s}.'.format(args.task))


def dgr_get_best_model_idx(model_type):
    if model_type == '24_medium':
        model_idx = 2
    elif model_type == '16_medium':
        model_idx = 4
    elif model_type == '8_large':
        model_idx = 2
    elif model_type == 'unet224':
        model_idx = 2
    elif model_type == 'unet448':
        model_idx = 2
    else:
        raise NotImplementedError
    return model_idx


if __name__ == "__main__":
    run()
