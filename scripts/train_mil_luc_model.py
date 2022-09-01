import argparse

from codecarbon import OfflineEmissionsTracker

from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
import dgr_luc_dataset
import dgr_luc_models

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('model', choices=['small', 'medium', 'resnet18'], help="Type of model to train.")
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    parser.add_argument('-t', '--track_emissions', action='store_true',
                        help='Whether or not to track emissions using CodeCarbon.')
    args = parser.parse_args()
    return args.model, args.n_repeats, args.track_emissions


def run_training():
    model_type, n_repeats, track_emissions = parse_args()

    tracker = None
    if track_emissions:
        tracker = OfflineEmissionsTracker(country_iso_code="GBR", project_name="Train_MIL_LUC_Model",
                                          output_dir="out/emissions", log_level='error')
        tracker.start()

    if model_type == 'small':
        dataset_clz = dgr_luc_dataset.DgrLucDatasetSmall
        model_clz = dgr_luc_models.DgrInstanceSpaceNNSmall
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)
    elif model_type == 'medium':
        dataset_clz = dgr_luc_dataset.DgrLucDatasetMedium
        model_clz = dgr_luc_models.DgrInstanceSpaceNNMedium
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)
    elif model_type == 'resnet18':
        dataset_clz = dgr_luc_dataset.DgrLucDatasetResNet18
        model_clz = dgr_luc_models.DgrResNet18
        trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, dataloader_func=create_normal_dataloader)
    else:
        raise ValueError("Training set up not provided for model type {:s}".format(model_type))

    dataset_name = dataset_clz.name
    model_name = model_clz.name

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_name)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(training_config, n_repeats=n_repeats)

    if tracker:
        tracker.stop()


if __name__ == "__main__":
    run_training()
