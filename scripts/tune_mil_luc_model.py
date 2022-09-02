import argparse

from codecarbon import OfflineEmissionsTracker

from bonfire.train.trainer import create_normal_dataloader
from bonfire.tune import create_tuner_from_config
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config
import dgr_luc_dataset
import dgr_luc_models

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC tuning script.')
    parser.add_argument('model', choices=['small', 'resnet18'], help="Type of model to tune.")
    parser.add_argument('-n', '--n_trials', default=70, type=int, help='The number of trials to run when tuning.')
    parser.add_argument('-t', '--track_emissions', action='store_true',
                        help='Whether or not to track emissions using CodeCarbon.')
    args = parser.parse_args()
    return args.model, args.n_trials, args.track_emissions


def run_tuning():
    model_type, n_trials, track_emissions = parse_args()

    tracker = None
    if track_emissions:
        tracker = OfflineEmissionsTracker(country_iso_code="GBR", project_name="Tune_MIL_LUC_Model",
                                          output_dir="out/emissions", log_level='error')
        tracker.start()

    if model_type == 'small':
        dataset_clz = dgr_luc_dataset.DgrLucDatasetSmall
        model_clz = dgr_luc_models.DgrInstanceSpaceNNSmall
    elif model_type == 'resnet18':
        dataset_clz = dgr_luc_dataset.DgrLucSingleInstanceDataset
        model_clz = dgr_luc_models.DgrResNet18
    else:
        raise ValueError("Training set up not provided for model type {:s}".format(model_type))

    dataset_name = dataset_clz.name
    model_name = model_clz.name

    # Create tuner
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    study_name = 'Tune_{:s}_{:s}'.format(dataset_name, model_name)
    tuner = create_tuner_from_config(device, model_clz, dataset_clz, config, study_name, n_trials,
                                     dataloader_func=create_normal_dataloader)

    # Log
    print('Starting {:s} tuning'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using dataset {:}'.format(dataset_name))
    print('  Using device {:}'.format(device))
    print('  Running study with {:d} trials'.format(n_trials))

    # Start run
    tuner.run_study()

    if tracker:
        tracker.stop()


if __name__ == "__main__":
    run_tuning()
