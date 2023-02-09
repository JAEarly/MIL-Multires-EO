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
    plotter.plot_grid_size_vs_performance()
    # plotter.plot_model_size_vs_performance()


class ResultsPlotter:

    def __init__(self):
        self.grid_sizes = {
            'S2P Single Res Small 8': 8,
            'S2P Single Res Medium 8': 8,
            'S2P Single Res Large 8': 8,
            'S2P Single Res Small 16': 16,
            'S2P Single Res Medium 16': 16,
            'S2P Single Res Large 16': 16,
            'S2P Single Res Small 32': 32,
            'S2P Single Res Medium 32': 32,
            'S2P Single Res Large 32': 32,
            'S2P Multi Res Single Out': 32,
            'S2P Multi Res Multi Out s=0': 8,
            'S2P Multi Res Multi Out s=1': 16,
            'S2P Multi Res Multi Out s=2': 32,
            'S2P Multi Res Multi Out s=m': 32
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
                if row[-5].strip() != 'N/A':
                    rmse_values.append(float(row[-5].strip().replace("\\textbf{", "")[:5]))
                    rmse_sems.append(float(row[-5].strip().replace("\\textbf{", "")[12:17]))
                else:
                    rmse_values.append(np.nan)
                    rmse_sems.append(np.nan)
                if row[-4].strip() != 'N/A':
                    mae_values.append(float(row[-4].strip().replace("\\textbf{", "")[:5]))
                    mae_sems.append(float(row[-4].strip().replace("\\textbf{", "")[12:17]))
                else:
                    mae_values.append(np.nan)
                    mae_sems.append(np.nan)
                if row[-2].strip() != 'N/A':
                    miou_values.append(float(row[-2].strip().replace("\\textbf{", "")[:5]))
                    miou_sems.append(float(row[-2].strip().replace("\\textbf{", "")[12:17]))
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
        if 'Single Res' in model_name:
            return self.colour_cycle[0]
        elif 'Single Out' in model_name:
            return self.colour_cycle[1]
        elif 'Multi Out' in model_name:
            if 's=m' in model_name:
                return self.colour_cycle[2]
            # If not the main output (s=m), include an alpha value to make the colour paler
            c_rgb = ImageColor.getcolor(self.colour_cycle[2], 'RGB')
            c1_rgba = (c_rgb[0]/255, c_rgb[1]/255, c_rgb[2]/255, 0.5)
            return c1_rgba
        raise ValueError('Colour not set for model name {:s}'.format(model_name))

    def get_models_for_grid_size_comparison(self, rmse_values_dict):
        best_8_model = np.argmin([rmse_values_dict[self.model_names[i]] for i in [0, 1, 2]])
        best_16_model = 3 + np.argmin([rmse_values_dict[self.model_names[i]] for i in [3, 4, 5]])
        best_32_model = 6 + np.argmin([rmse_values_dict[self.model_names[i]] for i in [6, 7, 8]])
        # Selected models are [best_8_model, s=0 MRMO, best_16_model, s=1 MRMO, best_32_model, s=2 MRMO, MRSO, s=m MRMO]
        model_idxs = [best_8_model, 10, best_16_model, 11, best_32_model, 12, 9, 13]
        return model_idxs

    def plot_grid_size_vs_performance(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4.3))

        self.plot_grid_size_vs_performance_single(axes[0], *self.dgr_results)
        self.plot_grid_size_vs_performance_single(axes[1], *self.floodnet_results)

        axes[0][0].set_title('RMSE', size=12)
        axes[0][1].set_title('MAE', size=12)
        axes[0][2].set_title('mIoU', size=12)
        for i in range(3):
            axes[1][i].set_xticks([0.5, 3.5, 7.5])
            axes[1][i].set_xticklabels(['8', '16', '32'])
            axes[1][i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[1][1].set_xlabel('Grid Size', size=13)
        axes[0][0].set_ylabel('DeepGlobe', size=14)
        axes[1][0].set_ylabel('FloodNet', size=14)

        labels = ['SR', 'MRMO s=\\{0,1,2\\}', 'MRSO', 'MRMO s=m']
        legend_colours = [self.model_colour(self.model_names[i]) for i in [0, 10, 9, 13]]
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in legend_colours]
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4, fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.198)

        save_path = "out/fig/s2p_grid_size_vs_performance.png".format()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_grid_size_vs_performance_single(self, axes, rmse_values_dict, rmse_sems_dict, mae_values_dict,
                                             mae_sems_dict, miou_values_dict, miou_sems_dict):
        xs = [0, 1, 3, 4, 6, 7, 8, 9]
        model_idxs = self.get_models_for_grid_size_comparison(rmse_values_dict)
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

    def plot_model_size_vs_performance(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4.3))

        self.plot_model_size_vs_performance_single(axes[0], *self.dgr_results)
        self.plot_model_size_vs_performance_single(axes[1], *self.floodnet_results)

        axes[0][0].set_title('RMSE', size=12)
        axes[0][1].set_title('MAE', size=12)
        axes[0][2].set_title('mIoU', size=12)
        for i in range(3):
            axes[1][i].set_xticks([1, 5, 9, 13])
            axes[1][i].set_xticklabels(['8', '16', '32', 'Avg'])
            axes[1][i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[1][1].set_xlabel('Grid Size', size=13)
        axes[0][0].set_ylabel('DeepGlobe', size=14)
        axes[1][0].set_ylabel('FloodNet', size=14)

        labels = ['Small', 'Medium', 'Large']
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in self.colour_cycle[:3] * 4]
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4, fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.198)

        save_path = "out/fig/s2p-sr_model_size_vs_performance.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_model_size_vs_performance_single(self, axes, rmse_values_dict, rmse_sems_dict, mae_values_dict,
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
