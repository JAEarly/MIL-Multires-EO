import argparse

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm

from bonfire.train.metrics import output_results, IoUMetric
from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device, load_model
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dataset import get_dataset_clz, get_model_type_list
from deepglobe import dgr_luc_dataset
from deepglobe.dgr_luc_models import get_model_clz as get_model_clz_dgr
from floodnet import floodnet_dataset
from floodnet.floodnet_models import get_model_clz as get_model_clz_floodnet

device = get_device()
dataset_choices = ["dgr", "floodnet"]


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset', choices=dataset_choices, help="Dataset to train on.")
    parser.add_argument('model_types', nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.dataset, args.model_types, args.n_repeats


def run_evaluation():
    wandb.init()

    dataset_name, model_types, n_repeats = parse_args()

    # Get list of datasets for the given dataset type
    if dataset_name == 'dgr':
        dataset_list = dgr_luc_dataset.get_dataset_list()
    elif dataset_name == 'floodnet':
        dataset_list = floodnet_dataset.get_dataset_list()
    else:
        raise ValueError('No dataset list for dataset {:s}'.format(dataset_name))

    # Get list of models to evaluate, either all or check that all requested models exist for the given dataset type
    model_type_choices = get_model_type_list(dataset_list)
    if model_types == ['all']:
        model_types = model_type_choices
    else:
        for model_type in model_types:
            if model_type not in model_type_choices:
                raise ValueError(
                    'Model type {:s} not found in model types for dataset {:s}'.format(model_type, dataset_name))

    # print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for models: {:}'.format(model_types))

    bag_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    grid_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    orig_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    for model_idx, model_type in enumerate(model_types):
        print('Evaluating {:s}'.format(model_type))

        # Get dataset clz
        dataset_clz = get_dataset_clz(model_type, dataset_list)

        # Get model clz
        if dataset_name == 'dgr':
            model_clz = get_model_clz_dgr(model_type)
        elif dataset_name == 'floodnet':
            model_clz = get_model_clz_floodnet(model_type)
        else:
            raise ValueError('No models for dataset {:s}'.format(dataset_name))

        config_path = "config/model_config.yaml"
        config = parse_yaml_config(config_path)
        training_config = parse_training_config(config['training'], model_type)
        wandb.config.update(training_config, allow_val_change=True)

        if 'resnet' in model_type or 'unet' in model_type or model_type == 'multi_res_single_out':
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz, dataloader_func=create_normal_dataloader)
        else:
            trainer = create_trainer_from_clzs(device, model_clz, dataset_clz)

        model_results = evaluate(model_type, n_repeats, trainer)
        model_bag_results, model_grid_seg_results, model_orig_seg_results = model_results
        bag_results[model_idx, :, :] = model_bag_results
        grid_seg_results[model_idx, :, :] = model_grid_seg_results
        orig_seg_results[model_idx, :, :] = model_orig_seg_results
    print('\nBag Results')
    output_results(model_types, bag_results, sort=False)
    print('\nGrid Segmentation Results')
    output_results(model_types, grid_seg_results, sort=False)
    print('\nHigh-Res Segmentation Results')
    output_results(model_types, orig_seg_results, sort=False, conf_mats=True)


def evaluate(model_type, n_repeats, trainer, random_state=5):
    bag_results_arr = np.empty((n_repeats, 3), dtype=object)
    grid_seg_results_arr = np.empty((n_repeats, 3), dtype=object)
    orig_seg_results_arr = np.empty((n_repeats, 3), dtype=object)

    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, False, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(device, trainer.dataset_clz.name, trainer.model_clz, modifier=r)

        results_list = eval_complete(model_type, trainer.metric_clz, model,
                                     train_dataloader, val_dataloader, test_dataloader, verbose=False, lim=None)

        train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res = results_list
        bag_results_arr[r, :] = [train_bag_res[0], val_bag_res[0], test_bag_res[0]]
        grid_seg_results_arr[r, :] = [train_inst_res[0], val_inst_res[0], test_inst_res[0]]
        orig_seg_results_arr[r, :] = [train_inst_res[1], val_inst_res[1], test_inst_res[1]]

        r += 1
        if r == n_repeats:
            break

    return bag_results_arr, grid_seg_results_arr, orig_seg_results_arr


def eval_complete(model_type, bag_metric, model, train_dataloader, val_dataloader, test_dataloader,
                  verbose=False, lim=None):
    train_bag_res, train_inst_res = eval_model(model_type, bag_metric, model, train_dataloader,
                                               verbose=verbose, lim=lim)
    val_bag_res, val_inst_res = eval_model(model_type, bag_metric, model, val_dataloader,
                                           verbose=verbose, lim=lim)
    test_bag_res, test_inst_res = eval_model(model_type, bag_metric, model, test_dataloader,
                                             verbose=verbose, lim=lim)
    return train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res


def eval_model(model_type, bag_metric, model, dataloader, verbose=False, lim=None):
    # Iterate through data loader and gather preds and targets
    all_preds = []
    all_targets = []
    all_instance_preds = []
    all_instance_targets = []
    all_metadatas = []
    labels = list(range(model.n_classes))
    model.eval()
    n = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Getting model predictions', leave=False):
            bags = data['bag']
            instance_targets = data['instance_targets']
            targets = data['target']
            metadata = data['bag_metadata']
            bag_pred, instance_pred = model.forward_verbose(bags, input_metadata=metadata)
            all_preds.append(bag_pred.cpu())
            all_targets.append(targets.cpu())
            all_metadatas.append(metadata)
            instance_pred = instance_pred[0]
            if instance_pred is not None:
                all_instance_preds.append(instance_pred.squeeze().cpu())
            all_instance_targets.append(instance_targets.squeeze().cpu())
            n += 1
            if lim is not None and n >= lim:
                break

    # Calculate bag results
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    bag_results = [bag_metric.calculate_metric(all_preds, all_targets, labels)]
    if verbose:
        for bag_result in bag_results:
            bag_result.out()

    # Calculate instance results
    grid_results = IoUMetric(torch.nan, torch.nan, None)
    seg_results = IoUMetric(torch.nan, torch.nan, None)
    if model_type != 'resnet':
        all_instance_preds = torch.stack(all_instance_preds)
        all_instance_targets = torch.stack(all_instance_targets)
        if 'unet' in model_type:
            # No evaluation for grid segmentation
            seg_results = evaluate_iou_segmentation(all_instance_preds, labels, all_metadatas,
                                                    dataloader.dataset.mask_img_to_clz_tensor)
        elif model_type != 'resnet':
            grid_results = evaluate_iou_grid(all_instance_preds, all_instance_targets, labels)
            seg_results = evaluate_iou_segmentation(all_instance_preds, labels, all_metadatas,
                                                    dataloader.dataset.mask_img_to_clz_tensor)

    instance_results = [grid_results, seg_results]
    if verbose:
        for instance_result in instance_results:
            instance_result.out()

    return bag_results, instance_results


def evaluate_iou_grid(grid_predictions, grid_targets, labels):
    # Evaluate IoU on grid preds and targets
    grid_clz_predictions = torch.argmax(grid_predictions, dim=1).long()
    grid_clz_targets = torch.argmax(grid_targets, dim=1).long()
    return IoUMetric.calculate_metric(grid_clz_predictions, grid_clz_targets, labels)


def evaluate_iou_segmentation(all_grid_predictions, labels, metadatas, mask_img_to_clz_tensor_func):
    """
    Evaluate IoU against original high res segmented images by scaling up the low resolution grid predictions
    """
    all_grid_clz_predictions = torch.argmax(all_grid_predictions, dim=1).long()

    # Compute IoU by first calculating the unions and intersections for every image, then doing a final computation
    # Storing all the predicted masks and true masks is too expensive
    all_conf_mats = []
    for idx, grid_clz_predictions in tqdm(enumerate(all_grid_clz_predictions), desc='Computing high res mIOU',
                                          leave=False, total=len(all_grid_clz_predictions)):
        # Load true mask image to compare to
        mask_img = Image.open(metadatas[idx]['mask_path'][0])
        mask_clz_tensor = mask_img_to_clz_tensor_func(mask_img)

        # Scale up grid predictions to same size as original image
        #  Have to double unsqueeze to add batch and channel dimensions so interpolation acts in the correct dimensions
        #  This is specific requirement of the F.interpolate method (requires batch and channel args)
        pred_clz_tensor = F.interpolate(grid_clz_predictions.float().unsqueeze(0).unsqueeze(0),
                                        size=mask_clz_tensor.shape, mode='nearest-exact')
        pred_clz_tensor = pred_clz_tensor.squeeze().long()

        # ----- Visual check -----
        # print(mask_clz_tensor.shape)
        # print(grid_clz_predictions.shape)
        # print(pred_clz_tensor.shape)
        # img = Image.open('data/FloodNet/train/train-org-img/{:d}.jpg'.format(metadatas[idx]['id'].item()))
        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        # axes[0][0].imshow(img)
        # axes[0][1].imshow(mask_img, cmap='tab10', vmin=0, vmax=10)
        # axes[0][2].imshow(mask_clz_tensor, cmap='tab10', vmin=0, vmax=10)
        # axes[1][1].imshow(pred_clz_tensor, cmap='tab10', vmin=0, vmax=10)
        # axes[1][2].imshow(grid_clz_predictions, cmap='tab10', vmin=0, vmax=10)
        # plt.tight_layout()
        # plt.show()
        # ----- Visual check -----

        # Compute intersection and union for this bag (used to calculate an overall IOU later)
        _, _, conf_mat = IoUMetric.intersection_over_union(mask_clz_tensor, pred_clz_tensor, len(labels))
        all_conf_mats.append(conf_mat)

    # Compute the final IoU score
    mean_iou, clz_iou, conf_mat = IoUMetric.calculate_from_cumulative(all_conf_mats)
    met = IoUMetric(mean_iou, clz_iou, conf_mat)
    return met


if __name__ == "__main__":
    run_evaluation()
