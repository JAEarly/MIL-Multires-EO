import torch
import wandb
from overrides import overrides
from torch import nn
from tqdm import tqdm

from bonfire.train.metrics import RegressionMetric
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
            bags = data['bag']
            targets = data['target'].to(self.device)
            metadata = data['bag_metadata']
            optimizer.zero_grad()
            outputs = model(bags, input_metadata=metadata)

            metric = self.metric_clz.calculate_metric(outputs, targets)
            loss = metric.rmse_loss

            loss.backward()
            optimizer.step()

            epoch_train_losses += metric.rmse_losses

        epoch_train_losses /= len(train_dataloader)
        epoch_train_metrics = self.metric_clz.from_train_loss(epoch_train_losses)
        epoch_val_metrics = None
        if val_dataloader is not None:
            bag_metrics, _ = self.eval_model(model, val_dataloader, bag_metrics=(self.metric_clz,))
            epoch_val_metrics = bag_metrics[0]

        return epoch_train_metrics, epoch_val_metrics

    @classmethod
    def get_model_outputs_for_dataset(cls, model, dataloader, lim=None):
        # Iterate through data loader and gather preds and targets
        all_preds = []
        all_targets = []
        all_instance_preds = []
        all_instance_targets = []
        all_metadatas = []
        model.eval()
        n = 0
        with torch.no_grad():
            for data in tqdm(dataloader, desc='Evaluating', leave=False):
                bags = data['bag']
                targets = data['target']
                instance_targets = data['instance_targets']
                metadata = data['bag_metadata']

                bag_pred, instance_pred = model.forward_verbose(bags, input_metadata=metadata)
                all_preds.append(bag_pred.cpu())
                all_targets.append(targets.cpu())
                all_metadatas.append(metadata)

                instance_pred = instance_pred[0]
                if instance_pred is not None:
                    all_instance_preds.append([i.squeeze().cpu() for i in instance_pred])
                all_instance_targets.append(instance_targets)
                n += 1
                if lim is not None and n >= lim:
                    break
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        return all_preds, all_targets, all_instance_preds, all_instance_targets, all_metadatas

    @classmethod
    @overrides
    def eval_model(cls, model, dataloader, bag_metrics=(), instance_metrics=(), verbose=False):
        # Iterate through data loader and gather preds and targets
        model.eval()
        model_outs = cls.get_model_outputs_for_dataset(model, dataloader)
        all_preds, all_targets, all_instance_preds, all_instance_targets = model_outs
        labels = list(range(model.n_classes))

        # Calculate bag results
        bag_results = None
        if bag_metrics:
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


class MultiResRegressionMetric(RegressionMetric):

    def __init__(self, overall_rmse_loss, overall_mae_loss, rmse_losses, mae_losses):
        super().__init__(overall_rmse_loss, overall_mae_loss)
        self.rmse_losses = rmse_losses
        self.mae_losses = mae_losses

    @staticmethod
    def calculate_metric(preds, targets, labels=None):
        rmse_losses = [MultiResRegressionMetric.criterion()(preds[:, i, :], targets) for i in range(4)]
        mae_losses = [nn.L1Loss()(preds[:, i, :].squeeze(), targets.squeeze()).item() for i in range(4)]
        avg_rmse_loss = sum(rmse_losses) / 4
        avg_mae_loss = sum(mae_losses) / 4
        # Only convert to tensors now to preserve gradients - if we convert then do the mean, gradients are lost
        rmse_losses = torch.as_tensor(rmse_losses)
        mae_losses = torch.as_tensor(mae_losses)
        return MultiResRegressionMetric(avg_rmse_loss, avg_mae_loss, rmse_losses, mae_losses)

    @staticmethod
    def from_train_loss(train_loss):
        overall_train_loss = sum(train_loss) / 4
        return MultiResRegressionMetric(overall_train_loss, None, train_loss, None)

    def short_string_repr(self):
        avg_rmse_str = "RMSE Loss: {:.3f}".format(self.rmse_loss.item())
        all_rmse_str = ", ".join(['{:.3f}'.format(l) for l in self.rmse_losses])
        avg_mae_str = "MAE Loss: {:.3f}".format(self.mae_loss) if self.mae_loss is not None else "MAE Loss: None"
        all_mae_str = ", ".join(['{:.3f}'.format(l) for l in self.mae_losses]) if self.mae_loss is not None else ""
        return '{:s} ({:s}); {:s} ({:s})'.format(avg_rmse_str, all_rmse_str, avg_mae_str, all_mae_str)

    def out(self):
        print('Average RMSE Loss: {:.3f}'.format(self.rmse_loss))
        print(' Scale RMSE Losses: ' + str(['{:.3f}'.format(l) for l in self.rmse_losses]))
        print('Average MAE Loss: {:.3f}'.format(self.mae_loss))
        print('  Scale MAE Losses: ' + str(['{:.3f}'.format(l) for l in self.mae_losses]))

    def wandb_log(self, dataset_split, commit):
        log_dict = {}
        if self.rmse_loss is not None:
            log_dict['{:s}_avg_rmse'.format(dataset_split)] = self.rmse_loss
        if self.rmse_losses is not None:
            log_dict['{:s}_s0_rmse'.format(dataset_split)] = self.rmse_losses[0]
            log_dict['{:s}_s1_rmse'.format(dataset_split)] = self.rmse_losses[1]
            log_dict['{:s}_s2_rmse'.format(dataset_split)] = self.rmse_losses[2]
            log_dict['{:s}_sm_rmse'.format(dataset_split)] = self.rmse_losses[3]
        if self.mae_loss is not None:
            log_dict['{:s}_avg_mae'.format(dataset_split)] = self.mae_loss
        if self.mae_losses is not None:
            log_dict['{:s}_s0_mae'.format(dataset_split)] = self.mae_losses[0]
            log_dict['{:s}_s1_mae'.format(dataset_split)] = self.mae_losses[1]
            log_dict['{:s}_s2_mae'.format(dataset_split)] = self.mae_losses[2]
            log_dict['{:s}_sm_mae'.format(dataset_split)] = self.mae_losses[3]
        wandb.log(log_dict, commit=commit)

    def wandb_summary(self, dataset_split):
        wandb.summary["{:s}_avg_rmse".format(dataset_split)] = self.rmse_loss
        wandb.summary['{:s}_s0_rmse'.format(dataset_split)] = self.rmse_losses[0]
        wandb.summary['{:s}_s1_rmse'.format(dataset_split)] = self.rmse_losses[1]
        wandb.summary['{:s}_s2_rmse'.format(dataset_split)] = self.rmse_losses[2]
        wandb.summary['{:s}_sm_rmse'.format(dataset_split)] = self.rmse_losses[3]
        wandb.summary["{:s}_avg_mae".format(dataset_split)] = self.mae_loss
        wandb.summary['{:s}_s0_mae'.format(dataset_split)] = self.mae_losses[0]
        wandb.summary['{:s}_s1_mae'.format(dataset_split)] = self.mae_losses[1]
        wandb.summary['{:s}_s2_mae'.format(dataset_split)] = self.mae_losses[2]
        wandb.summary['{:s}_sm_mae'.format(dataset_split)] = self.mae_losses[3]
