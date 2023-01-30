import argparse

from codecarbon import OfflineEmissionsTracker

from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dataset import get_dataset_clz, get_model_type_list
from dgr_luc_models import get_model_clz as get_model_clz_dgr
from floodnet.floodnet_models import get_model_clz as get_model_clz_floodnet
from dgr_luc_multires_trainer import MultiResTrainer
import dgr_luc_dataset
from floodnet import floodnet_dataset

device = get_device()
dataset_choices = ["dgr", "floodnet"]


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('dataset', choices=dataset_choices, help="Dataset to train on.")
    parser.add_argument('model', help="Type of model to train.")
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    parser.add_argument('-t', '--track_emissions', action='store_true',
                        help='Whether or not to track emissions using CodeCarbon.')
    args = parser.parse_args()
    return args.dataset, args.model, args.n_repeats, args.track_emissions


def run_training():
    dataset_name, model_type, n_repeats, track_emissions = parse_args()

    tracker = None
    if track_emissions:
        tracker = OfflineEmissionsTracker(country_iso_code="GBR", project_name="Train_MIL_LUC_Model",
                                          output_dir="out/emissions", log_level='error')
        tracker.start()

    if dataset_name == 'dgr':
        dataset_list = dgr_luc_dataset.get_dataset_list()
    elif dataset_name == 'floodnet':
        dataset_list = floodnet_dataset.get_dataset_list()
    else:
        raise ValueError('No dataset list for dataset {:s}'.format(dataset_name))

    model_type_choices = get_model_type_list(dataset_list)
    if model_type not in model_type_choices:
        raise ValueError('Model type {:s} not found in model types for dataset {:s}'.format(model_type, dataset_name))

    if dataset_name == 'dgr':
        model_clz = get_model_clz_dgr(model_type)
    elif dataset_name == 'floodnet':
        model_clz = get_model_clz_floodnet(model_type)
    else:
        raise ValueError('No models for dataset {:s}'.format(dataset_name))
    dataset_clz = get_dataset_clz(model_type, dataset_list)

    project_name = "Train_MIL_{:s}".format(dataset_name)
    group_name = "Train_{:s}".format(model_type)
    if 'resnet' in model_type or 'unet' in model_type or model_type == 'multi_res_single_out':
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, project_name=project_name,
                                           dataloader_func=create_normal_dataloader, group_name=group_name)
    elif model_type == 'multi_res_multi_out':
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, project_name=project_name,
                                           dataloader_func=create_normal_dataloader, group_name=group_name,
                                           trainer_clz=MultiResTrainer)
    else:
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz,
                                           project_name=project_name, group_name=group_name)

    # Parse wandb config and get training config for this model
    if dataset_name == 'dgr':
        config_path = "config/dgr_luc_config.yaml"
    elif dataset_name == 'floodnet':
        config_path = "config/floodnet_config.yaml"
    else:
        raise ValueError('No config file registered for dataset {:s}'.format(dataset_name))
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_type)

    # Log
    print('Starting {:s} training'.format(dataset_clz.name))
    print('  Using model {:} {:} and dataset {:}'.format(model_type, model_clz, dataset_clz))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(training_config, n_repeats=n_repeats)

    if tracker:
        tracker.stop()


if __name__ == "__main__":
    run_training()
