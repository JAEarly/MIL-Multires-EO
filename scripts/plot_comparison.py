import csv

import numpy as np
# Need to keep scienceplots imported for matplotlib styling even though the import is never used directly
# noinspection PyUnresolvedReferences
import scienceplots
from PIL import ImageColor
from matplotlib import pyplot as plt

plt.style.use(['science', 'bright'])


def run():
    plotter = ResultsPlotter()
    plotter.plot_main_results()
    plotter.plot_single_res_ablation_study()


class ResultsPlotter:

    def __init__(self):
        self.grid_sizes = {
            'S2P SR Small s=0': 8,
            'S2P SR Medium s=0': 8,
            'S2P SR Large s=0': 8,
            'S2P SR Small s=1': 16,
            'S2P SR Medium s=1': 16,
            'S2P SR Large s=1': 16,
            'S2P SR Small s=2': 32,
            'S2P SR Medium s=2': 32,
            'S2P SR Large s=2': 32,
            'S2P MRSO s=m': 32,
            'S2P MRMO s=0': 8,
            'S2P MRMO s=1': 16,
            'S2P MRMO s=2': 32,
            'S2P MRMO s=m': 32
        }
        self.model_names = list(self.grid_sizes.keys())
        self.dgr_results = self.parse_results("results/DeepGlobe/summarised_results_texttable_dgr.txt")
        self.floodnet_results = self.parse_results("results/FloodNet/summarised_results_texttable_floodnet.txt")
        self.colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def parse_results(self, results_file):
        with open(results_file, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter='|')
            rows = []
            for row in reader:
                rows.append(row)

            data = rows[4:38:2]
            model_names = [row[1].strip() for row in data]

            rmse_values = []
            rmse_sems = []
            mae_values = []
            mae_sems = []
            miou_values = []
            miou_sems = []

            for row in data:

                def split_result(entry):
                    s = entry.strip().replace("\\textbf{", "").replace("}", "").split(" $\pm$ ")
                    return float(s[0]), float(s[1])

                if row[-5].strip() != 'N/A':
                    mean, sem = split_result(row[-5])
                    rmse_values.append(mean)
                    rmse_sems.append(sem)
                else:
                    rmse_values.append(np.nan)
                    rmse_sems.append(np.nan)
                if row[-4].strip() != 'N/A':
                    mean, sem = split_result(row[-4])
                    mae_values.append(mean)
                    mae_sems.append(sem)
                else:
                    mae_values.append(np.nan)
                    mae_sems.append(np.nan)
                if row[-2].strip() != 'N/A':
                    mean, sem = split_result(row[-2])
                    miou_values.append(mean)
                    miou_sems.append(sem)
                else:
                    miou_values.append(np.nan)
                    miou_sems.append(np.nan)

            rmse_values_dict = dict(zip(model_names, rmse_values))
            rmse_sems_dict = dict(zip(model_names, rmse_sems))
            mae_values_dict = dict(zip(model_names, mae_values))
            mae_sems_dict = dict(zip(model_names, mae_sems))
            miou_values_dict = dict(zip(model_names, miou_values))
            miou_sems_dict = dict(zip(model_names, miou_sems))

            return rmse_values_dict, rmse_sems_dict, mae_values_dict, mae_sems_dict, miou_values_dict, miou_sems_dict

    def model_colour(self, model_name):
        if 'S2P SR' in model_name:
            return self.colour_cycle[0]
        elif 'S2P MRSO' in model_name:
            return self.colour_cycle[1]
        elif 'S2P MRMO' in model_name:
            return self.colour_cycle[2]
        raise ValueError('Colour not set for model name {:s}'.format(model_name))

    def plot_main_results(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4.3))

        self.plot_main_results_single(axes[0], *self.dgr_results)
        self.plot_main_results_single(axes[1], *self.floodnet_results)

        axes[0][0].set_title('Scene RMSE', size=12)
        axes[0][1].set_title('Scene MAE', size=12)
        axes[0][2].set_title('Pixel mIoU', size=12)
        for i in range(3):
            axes[1][i].set_xticks([0.5, 3.5, 6.5, 9.5])
            axes[1][i].set_xticklabels(['s = 0', 's = 1', 's = 2', 's = m'])
            axes[1][i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        # axes[1][1].set_xlabel('Grid Size', size=13)
        axes[0][0].set_ylabel('DeepGlobe', size=14)
        axes[1][0].set_ylabel('FloodNet', size=14)

        labels = ['Single Resolution', 'Multi-Resolution Single Out', 'Multi-Resolution Multi-Out']
        legend_colours = [self.model_colour(self.model_names[i]) for i in [0, 9, 13]]
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in legend_colours]
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4, fontsize=13)

        axes[0][0].set_ylim(0.075, 0.115)
        axes[0][0].set_yticks([0.075, 0.085, 0.095, 0.105, 0.115])
        axes[0][1].set_ylim(0.04, 0.06)
        axes[0][1].set_yticks([0.040, 0.045, 0.050, 0.055, 0.060])
        axes[0][2].set_ylim(0.25, 0.45)
        axes[0][2].set_yticks([0.25, 0.30, 0.35, 0.40, 0.45])
        axes[1][0].set_ylim(0.067, 0.075)
        axes[1][0].set_yticks([0.067, 0.069, 0.071, 0.073, 0.075])
        axes[1][1].set_ylim(0.024, 0.032)
        axes[1][1].set_yticks([0.024, 0.026, 0.028, 0.030, 0.032])
        axes[1][2].set_ylim(0.22, 0.28)
        axes[1][2].set_yticks([0.2, 0.22, 0.24, 0.26, 0.28])

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = "out/fig/s2p_main_results.png".format()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_main_results_single(self, axes, rmse_values_dict, rmse_sems_dict, mae_values_dict, mae_sems_dict,
                                 miou_values_dict, miou_sems_dict):
        xs = [0, 1, 3, 4, 6, 7, 9, 10]
        # Selected models are [8_large, s=0 MRMO, 16_large, s=1 MRMO, 32_large, s=2 MRMO, MRSO, s=m MRMO]
        model_idxs = [2, 10, 5, 11, 8, 12, 9, 13]
        rmse_ys = [rmse_values_dict[self.model_names[i]] for i in model_idxs]
        rmse_sems = [rmse_sems_dict[self.model_names[i]] for i in model_idxs]
        mae_ys = [mae_values_dict[self.model_names[i]] for i in model_idxs]
        mae_sems = [mae_sems_dict[self.model_names[i]] for i in model_idxs]
        miou_ys = [miou_values_dict[self.model_names[i]] for i in model_idxs]
        miou_sems = [miou_sems_dict[self.model_names[i]] for i in model_idxs]
        cs = [self.model_colour(self.model_names[i]) for i in model_idxs]

        axes[0].bar(xs, rmse_ys, color=cs, yerr=rmse_sems)
        axes[0].set_xticks([])
        axes[0].yaxis.tick_right()

        axes[1].bar(xs, mae_ys, color=cs, yerr=mae_sems)
        axes[1].set_xticks([])
        axes[1].yaxis.tick_right()

        axes[2].bar(xs, miou_ys, color=cs, yerr=miou_sems)
        axes[2].set_xticks([])
        axes[2].yaxis.tick_right()

    def plot_single_res_ablation_study(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4.3))

        self.plot_single_res_ablation_study_single(axes[0], *self.dgr_results)
        self.plot_single_res_ablation_study_single(axes[1], *self.floodnet_results)

        axes[0][0].set_title('Scene RMSE', size=12)
        axes[0][1].set_title('Scene MAE', size=12)
        axes[0][2].set_title('Pixel mIoU', size=12)
        for i in range(3):
            axes[1][i].set_xticks([1, 5, 9, 13])
            axes[1][i].set_xticklabels(['s = 0', 's = 1', 's = 2', 'Avg'])
            axes[1][i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        # axes[1][1].set_xlabel('Grid Size', size=13)
        axes[0][0].set_ylabel('DeepGlobe', size=14)
        axes[1][0].set_ylabel('FloodNet', size=14)

        axes[0][0].set_ylim(0.05, 0.13)
        axes[0][0].set_yticks([0.05, 0.07, 0.09, 0.11, 0.13])
        axes[0][1].set_ylim(0.03, 0.07)
        axes[0][1].set_yticks([0.03, 0.04, 0.05, 0.06, 0.07])
        axes[0][2].set_ylim(0.2, 0.42)
        axes[0][2].set_yticks([0.2, 0.26, 0.32, 0.38, 0.44])
        axes[1][0].set_ylim(0.06, 0.08)
        axes[1][0].set_yticks([0.06, 0.065, 0.07, 0.075, 0.08])
        axes[1][1].set_ylim(0.02, 0.036)
        axes[1][1].set_yticks([0.020, 0.024, 0.028, 0.032, 0.036])
        axes[1][2].set_ylim(0.15, 0.27)
        axes[1][2].set_yticks([0.15, 0.18, 0.21, 0.24, 0.27])

        labels = ['Small', 'Medium', 'Large']
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in self.colour_cycle[:3] * 4]
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4, fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = "out/fig/s2p-sr_ablation_study.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_single_res_ablation_study_single(self, axes, rmse_values_dict, rmse_sems_dict, mae_values_dict,
                                              mae_sems_dict, miou_values_dict, miou_sems_dict):
        model_names = self.grid_sizes.keys()
        xs = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        rmse_ys = [rmse_values_dict[model] for model in model_names][:9]
        rmse_sems = [rmse_sems_dict[model] for model in model_names][:9]
        mae_ys = [mae_values_dict[model] for model in model_names][:9]
        mae_sems = [mae_sems_dict[model] for model in model_names][:9]
        miou_ys = [miou_values_dict[model] for model in model_names][:9]
        miou_sems = [miou_sems_dict[model] for model in model_names][:9]
        cs = self.colour_cycle[:3] * 4

        for model_size_idx in range(3):
            model_idxs = [model_size_idx, 3 + model_size_idx, 6 + model_size_idx]
            rmse_ys.append(np.mean([rmse_ys[i] for i in model_idxs]))
            rmse_sems.append(np.mean([rmse_sems[i] for i in model_idxs]))
            mae_ys.append(np.mean([mae_ys[i] for i in model_idxs]))
            mae_sems.append(np.mean([mae_sems[i] for i in model_idxs]))
            miou_ys.append(np.mean([miou_ys[i] for i in model_idxs]))
            miou_sems.append(np.mean([miou_sems[i] for i in model_idxs]))

        axes[0].bar(xs, rmse_ys, color=cs, yerr=rmse_sems)
        axes[0].set_xticks([])
        axes[0].yaxis.tick_right()

        axes[1].bar(xs, mae_ys, color=cs, yerr=mae_sems)
        axes[1].set_xticks([])
        axes[1].yaxis.tick_right()

        axes[2].bar(xs, miou_ys, color=cs, yerr=miou_sems)
        axes[2].set_xticks([])
        axes[2].yaxis.tick_right()


if __name__ == "__main__":
    run()
