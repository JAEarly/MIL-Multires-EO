import csv

import matplotlib.pyplot as plt
# Need to keep scienceplots imported for matplotlib styling even though the import is never used directly
# noinspection PyUnresolvedReferences
import scienceplots
import seaborn as sns
import torch
import torch.nn.functional as f

plt.style.use(['science', 'bright'])


def run():
    floodnet_path = "results/FloodNet/conf_mats.txt"

    floodnet_model_names, floodnet_conf_mats = parse_conf_mats(floodnet_path, 10)
    plot_conf_mats(floodnet_model_names, floodnet_conf_mats)


def parse_conf_mats(path, n_classes):
    model_names = []
    conf_mats = []
    with open(path, 'r') as conf_mat_file:
        reader = csv.reader(conf_mat_file, delimiter='|')
        c = 0
        for line in reader:
            if c == 0:
                model_name = line[0]
                conf_mat = torch.zeros((n_classes, n_classes))
            else:
                if c % 2 == 0:
                    row = [float(l.strip()) for l in line[1:-1]]
                    conf_mat[(c - 1) // 2, :] = torch.as_tensor(row)
            c += 1
            if c == n_classes * 2 + 2:
                c = 0
                model_names.append(model_name)
                conf_mats.append(conf_mat)
    return model_names, conf_mats


def plot_conf_mats(floodnet_model_names, conf_mats):
    fig, axes = plt.subplots(nrows=3, ncols=len(conf_mats), figsize=(13, 6.5),
                             gridspec_kw={'height_ratios': [3.5, 1.5, 1.5]})

    for i, conf_mat in enumerate(conf_mats):
        # Row normalise
        row_norm_test_conf_mat = f.normalize(conf_mat, p=1, dim=1)
        plot_conf_mat(axes[0][i], row_norm_test_conf_mat)
        plot_per_class_precision(axes[1][i], conf_mat)
        plot_per_class_recall(axes[2][i], conf_mat)
        axes[0][i].set_title(floodnet_model_names[i], size=15)

    axes[0][0].set_ylabel('True Label', size=14)
    axes[1][0].set_ylabel('Precision', size=14)
    axes[2][0].set_ylabel('Recall', size=14)

    axes[0][2].set_xlabel('Predicted Label', size=13)
    axes[2][2].set_xlabel('Class', size=13)

    for i in range(len(conf_mats)):
        axes[1][i].set_xticklabels('')
        if i != 0:
            axes[1][i].set_yticklabels('')
            axes[2][i].set_yticklabels('')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.29, hspace=0.09)
    plt.show()


def plot_conf_mat(axis, conf_mat, vmax=None, label_x=True, label_y=True):
    sns.heatmap(conf_mat, ax=axis, fmt='', cbar=False, cmap='viridis',
                vmin=0, vmax=vmax if vmax is not None else torch.max(conf_mat))
    axis.set_aspect('equal')
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=label_x)
    axis.tick_params(axis='y', which='both', left=False, right=False, labelleft=label_y, labelright=False,
                     labelrotation=0)
    # axis.set_xlabel('Predicted Label', size=13)
    # axis.set_ylabel('True Label', size=14)


def plot_per_class_precision(axis, conf_mat):
    per_class_precision = torch.diag(conf_mat) / torch.sum(conf_mat, dim=0)
    per_class_precision = torch.nan_to_num(per_class_precision)
    avg_precision = torch.mean(per_class_precision)
    xs = list(range(len(per_class_precision)))
    axis.bar(xs, per_class_precision)
    axis.plot([-0.6, 9.6], [avg_precision] * 2, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], ls='--')
    axis.text(-0.31, 0.89, '{:.3f}'.format(avg_precision), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=True)
    axis.set_xticks(xs)
    # axis.set_xlabel('Class', size=13)
    # axis.set_ylabel('Recall', size=14)
    axis.set_xlim(-0.6, 9.6)
    axis.set_ylim(0, 1)


def plot_per_class_recall(axis, conf_mat):
    per_class_recall = torch.diag(conf_mat) / torch.sum(conf_mat, dim=1)
    avg_recall = torch.mean(per_class_recall)
    xs = list(range(len(per_class_recall)))
    axis.bar(xs, per_class_recall)
    axis.plot([-0.6, 9.6], [avg_recall] * 2, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], ls='--')
    axis.text(-0.3, 0.89, '{:.3f}'.format(avg_recall), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=True)
    axis.set_xticks(xs)
    # axis.set_xlabel('Class', size=13)
    # axis.set_ylabel('Recall', size=14)
    axis.set_xlim(-0.6, 9.6)
    axis.set_ylim(0, 1)


if __name__ == "__main__":
    run()
