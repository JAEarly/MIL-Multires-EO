import argparse

import wandb

from bonfire.util import get_device
from bonfire.util import load_model_from_path
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, DgrLucDataset
from dgr_luc_interpretability import MilLucInterpretabilityStudy
from dgr_luc_models import get_model_clz

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('task', choices=['reconstruct', 'interpret'], help='The task to perform.')
    args = parser.parse_args()
    return args.task


def run():
    task = parse_args()

    model_type = "8_large"
    if model_type == '24_medium':
        model_idx = 2
    elif model_type == '16_medium':
        model_idx = 4
    elif model_type == '8_large':
        model_idx = 2
    else:
        raise NotImplementedError
    model_clz = get_model_clz(model_type)
    dataset_clz = get_dataset_clz(model_type)

    model_path = "models/dgr_luc_{:s}/InstanceSpaceNN/InstanceSpaceNN_{:d}.pkl".format(model_type, model_idx)

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_type)
    wandb.init(
        config=training_config,
    )

    complete_dataset = dataset_clz.create_complete_dataset()
    model = load_model_from_path(device, model_clz, model_path)
    study = MilLucInterpretabilityStudy(device, complete_dataset, model)

    if task == 'reconstruct':
        study.create_reconstructions()
    elif task == 'interpret':
        study.sample_interpretations()
        # study.create_interpretation_from_id(739760)
    else:
        raise NotImplementedError('Task not implemented: {:s}.'.format(task))


if __name__ == "__main__":
    run()
