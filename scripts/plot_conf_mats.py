import csv

import matplotlib.pyplot as plt
# Need to keep scienceplots imported for matplotlib styling even though the import is never used directly
# noinspection PyUnresolvedReferences
import scienceplots
import seaborn as sns
import torch
import torch.nn.functional as f
from matplotlib import cm

plt.style.use(['science', 'bright'])


def run(dataset_name, path, n_classes, clz_names, avg_label_loc):
    print('Plotting conf mats from {:s}'.format(path))
    model_names, conf_mats = parse_conf_mats(path, n_classes)
    plot_conf_mats(dataset_name, model_names, conf_mats, clz_names, avg_label_loc)


def parse_conf_mats(path, n_classes):
    print('Parsing data...')
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


def plot_conf_mats(dataset_name, floodnet_model_names, conf_mats, clz_names, avg_label_loc):
    print('Plotting...')
    fig, axes = plt.subplots(nrows=3, ncols=len(conf_mats) + 1, figsize=(13, 6.5),
                             gridspec_kw={'height_ratios': [3.5, 1.5, 1.5],
                                          'width_ratios': [1] * len(conf_mats) + [0.1]})

    for i, conf_mat in enumerate(conf_mats):
        # Row normalise
        plot_conf_mat(axes[0][i], conf_mat)
        plot_per_class_precision(axes[1][i], conf_mat, avg_label_loc)
        plot_per_class_recall(axes[2][i], conf_mat, avg_label_loc)
        axes[0][i].set_title(floodnet_model_names[i], size=15)

    axes[0][0].set_ylabel('True Label', size=14)
    axes[1][0].set_ylabel('Precision', size=14)
    axes[2][0].set_ylabel('Recall', size=14)

    axes[0][2].set_xlabel('Predicted Label', size=13)
    axes[2][2].set_xlabel('Class', size=13)

    axes[0][-1].set_aspect(9.5)
    fig.colorbar(cm.ScalarMappable(norm=None, cmap='viridis'), cax=axes[0][-1])

    axes[1][-1].set_axis_off()
    axes[2][-1].set_axis_off()

    for i in range(5):
        axes[1][i].set_xticklabels('')

    x = 0.885
    y = 0.5
    plt.text(x, y, r"\textbf{Classes}", transform=plt.gcf().transFigure, size=13,
             horizontalalignment='left', verticalalignment='top')
    y -= 0.03
    for idx, clz_name in enumerate(clz_names):
        plt.text(x, y, '{:d}:'.format(idx),
                 transform=plt.gcf().transFigure, size=12, horizontalalignment='left', verticalalignment='top')
        plt.text(x + 0.015, y, '{:s}'.format(clz_name),
                 transform=plt.gcf().transFigure, size=12, horizontalalignment='left', verticalalignment='top')
        y -= 0.055 if '\n' in clz_name or '\\' in clz_name else 0.03
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.29, hspace=0.09)
    save_path = "out/fig/conf_mats_{:s}.png".format(dataset_name)
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_conf_mat(axis, conf_mat, vmax=None, label_x=True, label_y=True):
    norm_conf_mat = f.normalize(conf_mat, p=1, dim=1)
    sns.heatmap(norm_conf_mat, ax=axis, fmt='', cbar=False, cmap='viridis',
                vmin=0, vmax=vmax if vmax is not None else torch.max(norm_conf_mat))
    axis.set_aspect('equal')
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=label_x)
    axis.tick_params(axis='y', which='both', left=False, right=False, labelleft=label_y, labelright=False,
                     labelrotation=0)


def plot_per_class_precision(axis, conf_mat, avg_label_loc):
    per_class_precision = torch.diag(conf_mat) / torch.sum(conf_mat, dim=0)
    per_class_precision = torch.nan_to_num(per_class_precision)
    avg_precision = torch.mean(per_class_precision)
    xs = list(range(len(per_class_precision)))
    axis.bar(xs, per_class_precision)
    axis.plot([-0.6, 9.6], [avg_precision] * 2, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], ls='--')
    if avg_label_loc == 'upper right':
        axis.text(0.98, 0.97, '{:.3f}'.format(avg_precision), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                  ha='right', va='top', transform=axis.transAxes)
    elif avg_label_loc == 'upper left':
        axis.text(0.02, 0.97, '{:.3f}'.format(avg_precision), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                  ha='left', va='top', transform=axis.transAxes)
    else:
        raise NotImplementedError
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=True)
    axis.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
    axis.set_xticks(xs)
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_xlim(-0.6, len(xs) - 0.4)
    axis.set_ylim(0, 1)


def plot_per_class_recall(axis, conf_mat, avg_label_loc):
    per_class_recall = torch.diag(conf_mat) / torch.sum(conf_mat, dim=1)
    avg_recall = torch.mean(per_class_recall)
    xs = list(range(len(per_class_recall)))
    axis.bar(xs, per_class_recall)
    axis.plot([-0.6, 9.6], [avg_recall] * 2, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], ls='--')
    if avg_label_loc == 'upper right':
        axis.text(0.98, 0.97, '{:.3f}'.format(avg_recall), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                  ha='right', va='top', transform=axis.transAxes)
    elif avg_label_loc == 'upper left':
        axis.text(0.02, 0.97, '{:.3f}'.format(avg_recall), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                  ha='left', va='top', transform=axis.transAxes)
    else:
        raise NotImplementedError
    axis.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=True)
    axis.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
    axis.set_xticks(xs)
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_xlim(-0.6, len(xs) - 0.4)
    axis.set_ylim(0, 1)


if __name__ == "__main__":
    run("DeepGlobe", "results/DeepGlobe/conf_mats.txt", 7,
        ['Urban', 'Agricultural', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown'],
        "upper right")
    run("FloodNet", "results/FloodNet/conf_mats.txt", 10,
        ['Background', 'Building\n Flooded', 'Building\n Non-flooded', 'Road\n Flooded', 'Road\n Non-Flooded', 'Water',
         'Tree', 'Vehicle', 'Pool', 'Grass'],
        "upper left")
