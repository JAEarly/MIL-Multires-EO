import torch
import wandb
from overrides import overrides
from torch import nn
from tqdm import tqdm

from bonfire.train.metrics import Metric
from bonfire.train.trainer import Trainer


class MultiResTrainer(Trainer):

    @property
    @overrides
    def metric_clz(self):
        return MultiResRegressionMetric

    @overrides
    def train_epoch(self, model, optimizer, criterion, train_dataloader, val_dataloader):
        model.train()
        epoch_train_losses = torch.zeros(4)
        for data in tqdm(train_dataloader, desc='Epoch Progress', leave=False):
            torch.cuda.empty_cache()
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)

            metric = self.metric_clz.calculate_metric(outputs, targets)
            loss = metric.overall_rmse_loss

            loss.backward()
            optimizer.step()

            epoch_train_losses += metric.rmse_losses

        epoch_train_losses /= len(train_dataloader)
        epoch_train_metrics = self.metric_clz.from_train_loss(epoch_train_losses)
        epoch_val_metrics = None
        if val_dataloader is not None:
            bag_metrics, _ = eval_multires_model(model, val_dataloader, bag_metrics=(self.metric_clz,))
            epoch_val_metrics = bag_metrics[0]

        return epoch_train_metrics, epoch_val_metrics


class MultiResRegressionMetric(Metric):

    optimise_direction = 'minimize'

    def __init__(self, overall_rmse_loss, overall_mae_loss, rmse_losses, mae_losses):
        self.overall_rmse_loss = overall_rmse_loss
        self.overall_mae_loss = overall_mae_loss
        self.rmse_losses = rmse_losses
        self.mae_losses = mae_losses

    def key_metric(self):
        return self.overall_rmse_loss

    @staticmethod
    def criterion():
        return lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs.squeeze(), targets.squeeze()))

    @staticmethod
    def calculate_metric(preds, targets, labels=None):
        rmse_losses = [MultiResRegressionMetric.criterion()(preds[:, i, :], targets) for i in range(4)]
        mae_losses = [nn.L1Loss()(preds[:, i, :].squeeze(), targets.squeeze()).item() for i in range(4)]
        overall_rmse_loss = sum(rmse_losses) / 4
        overall_mae_loss = sum(mae_losses) / 4
        # Only convert to tensors now to preserve gradients - if we convert then do the mean, gradients are lost
        rmse_losses = torch.as_tensor(rmse_losses)
        mae_losses = torch.as_tensor(mae_losses)
        return MultiResRegressionMetric(overall_rmse_loss, overall_mae_loss, rmse_losses, mae_losses)

    @staticmethod
    def from_train_loss(train_loss):
        overall_train_loss = sum(train_loss) / 4
        return MultiResRegressionMetric(overall_train_loss, None, train_loss, None)

    def short_string_repr(self):
        return "{{Overall RMSE Loss: {:.3f}; ".format(self.overall_rmse_loss.item()) + \
               ("Overall MAE Loss: {:.3f}}}".format(self.overall_mae_loss)
                if self.overall_mae_loss is not None else "MAE Loss: None}")

    def out(self):
        print('Overall RMSE Loss: {:.3f}'.format(self.overall_rmse_loss))
        print('Overall MAE Loss: {:.3f}'.format(self.overall_mae_loss))

    def wandb_log(self, dataset_split, commit):
        log_dict = {}
        if self.overall_rmse_loss is not None:
            log_dict['{:s}_rmse'.format(dataset_split)] = self.overall_rmse_loss
        if self.overall_mae_loss is not None:
            log_dict['{:s}_mae'.format(dataset_split)] = self.overall_mae_loss
        wandb.log(log_dict, commit=commit)

    def wandb_summary(self, dataset_split):
        wandb.summary["{:s}_rmse".format(dataset_split)] = self.overall_rmse_loss
        wandb.summary["{:s}_mae".format(dataset_split)] = self.overall_mae_loss


def eval_multires_model(model, dataloader, bag_metrics=(), instance_metrics=(), verbose=False):
    # Iterate through data loader and gather preds and targets
    all_preds = []
    all_targets = []
    all_instance_preds = []
    all_instance_targets = []
    labels = list(range(model.n_classes))
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating', leave=False):
            bags, targets, instance_targets = data[0], data[1], data[2]
            bag_pred, instance_pred = model.forward_verbose(bags)
            all_preds.append(bag_pred.cpu())
            all_targets.append(targets.cpu())

            instance_pred = instance_pred[0]
            if instance_pred is not None:
                all_instance_preds.append([i.squeeze().cpu() for i in instance_pred])
            all_instance_targets.append(instance_targets.squeeze().cpu())

    # Calculate bag results
    bag_results = None
    if bag_metrics:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        bag_results = [bm.calculate_metric(all_preds, all_targets, labels) for bm in bag_metrics]
        if verbose:
            for bag_result in bag_results:
                bag_result.out()

    # Calculate instance results
    instance_results = None
    if instance_metrics:
        all_instance_preds = torch.cat(all_instance_preds)
        all_instance_targets = torch.cat(all_instance_targets)
        instance_results = [im.calculate_metric(all_instance_preds, all_instance_targets, labels)
                            for im in instance_metrics]
        if verbose:
            for instance_result in instance_results:
                instance_result.out()

    return bag_results, instance_results

