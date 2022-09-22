import argparse

import numpy as np
import torch
import wandb
from PIL import Image
from tqdm import tqdm

from bonfire.train.metrics import output_results, IoUMetric
from bonfire.train.trainer import create_trainer_from_clzs, create_normal_dataloader
from bonfire.util import get_device, load_model
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config
from dgr_luc_dataset import get_dataset_clz, get_model_type_list, get_patch_details, make_binary_mask, DgrLucDataset
from dgr_luc_models import get_model_clz

device = get_device()
all_models = get_model_type_list()
model_type_choices = all_models + ['all']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('model_types', choices=model_type_choices, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    parser.add_argument('-s', '--evaluate_segmentation', action='store_true', help='If to evaluate segmentation.')
    args = parser.parse_args()
    return args.model_types, args.n_repeats, args.evaluate_segmentation


def run_evaluation():
    wandb.init()

    model_types, n_repeats, evaluate_segmentation = parse_args()

    if model_types == ['all']:
        model_types = all_models

    # print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for models: {:}'.format(model_types))

    bag_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    inst_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    grid_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
    orig_seg_results = np.empty((len(model_types), n_repeats, 3), dtype=object)
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

        model_results = evaluate(model_type, n_repeats, trainer, evaluate_segmentation=evaluate_segmentation)
        model_bag_results, model_inst_results, model_grid_seg_results, model_orig_seg_results = model_results
        bag_results[model_idx, :, :] = model_bag_results
        inst_results[model_idx, :, :] = model_inst_results
        grid_seg_results[model_idx, :, :] = model_grid_seg_results
        orig_seg_results[model_idx, :, :] = model_orig_seg_results
    print('\nBag Results')
    output_results(model_types, bag_results, sort=False)
    print('\nInstance Results')
    output_results(model_types, inst_results, sort=False)
    print('\nGrid Segmentation Results')
    output_results(model_types, grid_seg_results, sort=False)
    print('\nHigh-Res Segmentation Results')
    output_results(model_types, orig_seg_results, sort=False)


def evaluate(model_type, n_repeats, trainer, random_state=5, evaluate_segmentation=False):
    bag_results_arr = np.empty((n_repeats, 3), dtype=object)
    inst_results_arr = np.empty((n_repeats, 3), dtype=object)
    grid_seg_results_arr = np.empty((n_repeats, 3), dtype=object)
    orig_seg_results_arr = np.empty((n_repeats, 3), dtype=object)

    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, True, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(device, trainer.dataset_clz.name, trainer.model_clz, modifier=r)

        results_list = eval_complete(model_type, trainer.metric_clz, model,
                                     train_dataloader, val_dataloader, test_dataloader,
                                     verbose=False, evaluate_segmentation=evaluate_segmentation)

        train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res = results_list
        bag_results_arr[r, :] = [train_bag_res[0], val_bag_res[0], test_bag_res[0]]
        if model_type != 'resnet':
            inst_results_arr[r, :] = [train_inst_res[0], val_inst_res[0], test_inst_res[0]]
            grid_seg_results_arr[r, :] = [train_inst_res[1], val_inst_res[1], test_inst_res[1]]
            orig_seg_results_arr[r, :] = [train_inst_res[2], val_inst_res[2], test_inst_res[2]]
        else:
            inst_results_arr[r, :] = trainer.metric_clz(torch.nan, torch.nan)
            grid_seg_results_arr[r, :] = IoUMetric(torch.nan, None)
            orig_seg_results_arr[r, :] = IoUMetric(torch.nan, None)

        r += 1
        if r == n_repeats:
            break

    return bag_results_arr, inst_results_arr, grid_seg_results_arr, orig_seg_results_arr


def eval_complete(model_type, bag_metric, model, train_dataloader, val_dataloader, test_dataloader,
                  verbose=False, evaluate_segmentation=False):
    train_bag_res, train_inst_res = eval_model(model_type, bag_metric, model, train_dataloader,
                                               verbose=verbose, evaluate_segmentation=evaluate_segmentation)
    val_bag_res, val_inst_res = eval_model(model_type, bag_metric, model, val_dataloader,
                                           verbose=verbose, evaluate_segmentation=evaluate_segmentation)
    test_bag_res, test_inst_res = eval_model(model_type, bag_metric, model, test_dataloader,
                                             verbose=verbose, evaluate_segmentation=evaluate_segmentation)
    return train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res


def eval_model(model_type, bag_metric, model, dataloader, verbose=False, evaluate_segmentation=False):
    # Iterate through data loader and gather preds and targets
    all_preds = []
    all_targets = []
    all_instance_preds = []
    all_instance_targets = []
    all_mask_paths = []
    labels = list(range(model.n_classes))
    model.eval()
    with torch.no_grad():
        i = 0
        for data in tqdm(dataloader, desc='Getting model predictions', leave=False):
            bags, targets, instance_targets, mask_path = data[0], data[1], data[2], data[3]
            bag_pred, instance_pred = model.forward_verbose(bags)
            all_preds.append(bag_pred.cpu())
            all_targets.append(targets.cpu())
            all_mask_paths.append(mask_path[0])

            instance_pred = instance_pred[0]
            if instance_pred is not None:
                all_instance_preds.append(instance_pred.squeeze().cpu())
            all_instance_targets.append(instance_targets.squeeze().cpu())
            if i > 100:
                break
            i += 1

    # Calculate bag results
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    bag_results = [bag_metric.calculate_metric(all_preds, all_targets, labels)]
    if verbose:
        for bag_result in bag_results:
            bag_result.out()

    # Calculate instance results
    instance_results = None
    if model_type != 'resnet':
        all_instance_preds = torch.cat(all_instance_preds)
        all_instance_targets = torch.cat(all_instance_targets)
        patch_details = get_patch_details(model_type)
        instance_main_results = bag_metric.calculate_metric(all_instance_preds, all_instance_targets, labels)
        if evaluate_segmentation:
            iou_grid_results = evaluate_iou_grid(all_instance_preds, all_instance_targets, labels,
                                                 patch_details.grid_size)
            iou_orig_results = evaluate_iou_orig(all_instance_preds, labels, patch_details.grid_size,
                                                 patch_details.cell_size, all_mask_paths)
            instance_results = [instance_main_results, iou_grid_results, iou_orig_results]
        else:
            instance_results = [instance_main_results, IoUMetric(torch.nan, None), IoUMetric(torch.nan, None)]
        if verbose:
            for instance_result in instance_results:
                instance_result.out()

    return bag_results, instance_results


def evaluate_iou_grid(predictions, targets, labels, grid_size):
    # Evaluate IoU on low res grid preds and targets
    grid_predictions = predictions.view(-1, grid_size, grid_size, len(labels))
    grid_targets = targets.view(-1, grid_size, grid_size, len(labels))
    grid_clz_predictions = torch.argmax(grid_predictions, dim=3).long()
    grid_clz_targets = torch.argmax(grid_targets, dim=3).long()
    met = IoUMetric.calculate_metric(grid_clz_predictions, grid_clz_targets, labels)
    return met


def evaluate_iou_orig(predictions, labels, grid_size, cell_size, mask_paths):
    # Evaluate IoU against original high res segmented images by scaling up the low resolution grid predictions
    grid_predictions = predictions.view(-1, grid_size, grid_size, len(labels))
    grid_clz_predictions = torch.argmax(grid_predictions, dim=3).long()

    # Compute IoU by first calculating the unions and intersections for every image, then doing a final computation
    # Storing all the predicted masks and true masks is too expensive
    all_conf_mats = []
    for idx in tqdm(range(len(grid_clz_predictions)), desc='Computing high res mIOU', leave=False):
        # Load true mask image to compare to
        mask_img = torch.as_tensor(np.array(Image.open(mask_paths[idx])))

        # Scale up grid predictions to same size as original image
        pred_img = torch.zeros_like(mask_img)
        for i_x in range(grid_size):
            for i_y in range(grid_size):
                cell_pred = grid_clz_predictions[idx, i_x, i_y].item()
                # print(cell_pred)
                cell_colour = [c * 255 for c in DgrLucDataset.target_to_rgb(cell_pred)]
                cell_colour = torch.as_tensor(cell_colour)
                # print(cell_colour)
                c_x = i_x * cell_size
                c_y = i_y * cell_size
                pred_img[c_x:c_x + cell_size, c_y:c_y + cell_size] = cell_colour

        # Convert coloured masked to binary (threshold at 128)
        mask_binary = make_binary_mask(mask_img)
        pred_binary = make_binary_mask(pred_img)

        # Convert threshold images to integer values (binary -> int conversion making use of ldexp)
        #  E.g., binary label of [1, 1, 0] -> [1, 2, 0] (ldexp) -> 3 (sum)
        #  NOTE: These derived labels are not the same as the original class labels, but it is still a valid mapping.
        #        We just need the colours to efficiently map to different integers for the IoU calculation
        #        Colours -> Threshold (binary) -> Integer
        #        We remap these classes below
        mask_clz_labels = torch.sum(torch.ldexp(mask_binary, torch.tensor([0, 1, 2])), dim=2).long()
        pred_clz_labels = torch.sum(torch.ldexp(pred_binary, torch.tensor([0, 1, 2])), dim=2).long()

        # Due to how the colours were picked for the classes (in the original dataset), nothing maps to one.
        #  I.e., the output above maps to 0,2,3,4,5,6,7
        # Therefore, subtract 1 from non-zero labels to get in the range 0 to 6.
        mask_clz_labels = torch.where(mask_clz_labels > 0, mask_clz_labels - 1, 0).long()
        pred_clz_labels = torch.where(pred_clz_labels > 0, pred_clz_labels - 1, 0).long()

        # Compute intersection and union for this bag (used to calculate an overall IOU later)
        _, _, conf_mat = IoUMetric.intersection_over_union(mask_clz_labels, pred_clz_labels, len(labels))

        # Remap the classes
        #   clz                        ->  binary   ->  n  ->  n + 1
        #   clz 0 (urban land)         ->  0, 1, 1  ->  5  ->  6
        #   clz 1 (agricultural land)  ->  1, 1, 0  ->  2  ->  3
        #   clz 2 (rangeland)          ->  1, 0, 1  ->  4  ->  5
        #   clz 3 (forest land)        ->  0, 1, 0  ->  1  ->  2
        #   clz 4 (water)              ->  0, 0, 1  ->  3  ->  4
        #   clz 5 (barren land)        ->  1, 1, 1  ->  6  ->  7
        #   clz 6 (unknown)            ->  0, 0, 0  ->  0  ->  0
        remap = [5, 2, 4, 1, 3, 6, 0]
        conf_mat = conf_mat[:, remap]
        conf_mat = conf_mat[remap, :]
        all_conf_mats.append(conf_mat)

        if idx == 10:
            break

    # Compute the final IoU score
    mean_iou, clz_iou, conf_mat = IoUMetric.calculate_from_cumulative(all_conf_mats)
    met = IoUMetric(mean_iou, clz_iou, conf_mat)
    return met


if __name__ == "__main__":
    run_evaluation()
