import argparse

import wandb

from bonfire.util import get_device
from bonfire.util import load_model_from_path
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import DgrLucDataset
from dgr_luc_interpretability import MilLucInterpretabilityStudy
from dgr_luc_models import DgrInstanceSpaceNN

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL LUC training script.')
    parser.add_argument('task', choices=['reconstruct', 'interpret'], help='The task to perform.')
    args = parser.parse_args()
    return args.task


def run():
    task = parse_args()

    model_clz = DgrInstanceSpaceNN
    model_name = model_clz.name
    model_path = "models/dgr_luc/InstanceSpaceNN/InstanceSpaceNN_0.pkl"

    # Parse wandb config and get training config for this model
    config_path = "config/dgr_luc_config.yaml"
    config = parse_yaml_config(config_path)
    training_config = parse_training_config(config['training'], model_name)
    wandb.init(
        config=training_config,
    )

    complete_dataset = DgrLucDataset.create_complete_dataset()
    model = load_model_from_path(device, model_clz, model_path)
    study = MilLucInterpretabilityStudy(device, complete_dataset, model)

    if task == 'reconstruct':
        study.create_reconstructions()
    elif task == 'interpret':
        study.create_interpretation_from_id(739760)
    else:
        raise NotImplementedError('Task not implemented: {:s}.'.format(task))


if __name__ == "__main__":
    run()
