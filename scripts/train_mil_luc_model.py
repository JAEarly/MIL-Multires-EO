import argparse

from codecarbon import OfflineEmissionsTracker

from bonfire.train.trainer import create_trainer_from_clzs
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import DgrLucDataset
from dgr_luc_models import DgrInstanceSpaceNN

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('-r', '--n_repeats', default=5, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.n_repeats


def run_training():
    dataset_clz = DgrLucDataset
    dataset_name = dataset_clz.name
    model_clz = DgrInstanceSpaceNN
    model_name = model_clz.name

    # Parse args
    n_repeats = parse_args()

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_name)

    # Create trainer
    trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(training_config, n_repeats=n_repeats)


if __name__ == "__main__":
    _tracker = OfflineEmissionsTracker(country_iso_code="GBR", project_name="Train_MIL_LUC_Model",
                                       output_dir="out/emissions", log_level='error')
    _tracker.start()
    run_training()
    _tracker.stop()
