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
    plotter.plot_model_size_vs_performance()


class ResultsPlotter:

    def __init__(self):
        self.grid_sizes = {
            'S2P Single Res Small 8': 8,
            'S2P Single Res Medium 8': 8,
            'S2P Single Res Large 8': 8,
            'S2P Single Res Small 16': 16,
            'S2P Single Res Medium 16': 16,
            'S2P Single Res Large 16': 16,
            'S2P Single Res Small 24': 24,
            'S2P Single Res Medium 24': 24,
            'S2P Single Res Large 24': 24,
            'S2P Multi Res Single Out': 32,
            'S2P Multi Res Multi Out s=0': 8,
            'S2P Multi Res Multi Out s=1': 16,
            'S2P Multi Res Multi Out s=2': 32,
            'S2P Multi Res Multi Out s=m': 32
        }
        self.model_names = list(self.grid_sizes.keys())
        self.results_files = "results/summarised_results_texttable.txt"
        self.rmse_values_dict, self.rmse_stds_dict, self.mae_values_dict,\
            self.mae_stds_dict, self.miou_values_dict, self.miou_stds_dict = self.parse_results()
        self.colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def parse_results(self):
        with open(self.results_files, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter='|')
            rows = []
            for row in reader:
                rows.append(row)

            data = rows[3::2]
            model_names = [row[1].strip() for row in data]

            rmse_values = []
            rmse_stds = []
            mae_values = []
            mae_stds = []
            miou_values = []
            miou_stds = []

            for row in data:
                rmse_values.append(float(row[-5].strip().replace("\\textbf{", "")[:5]))
                rmse_stds.append(float(row[-5].strip().replace("\\textbf{", "")[12:17]))
                mae_values.append(float(row[-4].strip().replace("\\textbf{", "")[:5]))
                mae_stds.append(float(row[-4].strip().replace("\\textbf{", "")[12:17]))
                if row[-2].strip() != 'N/A':
                    miou_values.append(float(row[-2].strip().replace("\\textbf{", "")[:5]))
                    miou_stds.append(float(row[-2].strip().replace("\\textbf{", "")[12:17]))
                else:
                    miou_values.append('N/A')
                    miou_stds.append('N/A')

            rmse_values_dict = dict(zip(model_names, rmse_values))
            rmse_stds_dict = dict(zip(model_names, rmse_stds))
            mae_values_dict = dict(zip(model_names, mae_values))
            mae_stds_dict = dict(zip(model_names, mae_stds))
            miou_values_dict = dict(zip(model_names, miou_values))
            miou_stds_dict = dict(zip(model_names, miou_stds))

            return rmse_values_dict, rmse_stds_dict, mae_values_dict, mae_stds_dict, miou_values_dict, miou_stds_dict

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

    def get_models_for_grid_size_comparison(self):
        best_8_model = np.argmin([self.rmse_values_dict[self.model_names[i]] for i in [0, 1, 2]])
        best_16_model = 3 + np.argmin([self.rmse_values_dict[self.model_names[i]] for i in [3, 4, 5]])
        # TODO this should be best 32 model when they're trained
        best_24_model = 6 + np.argmin([self.rmse_values_dict[self.model_names[i]] for i in [6, 7, 8]])
        # Selected models are [best_8_model, s=0 MRMO, best_16_model, s=1 MRMO, best_24_model, s=2 MRMO, s=m MRMO, MRSO]
        model_idxs = [best_8_model, 10, best_16_model, 11, best_24_model, 12, 13, 9]
        return model_idxs

    def plot_grid_size_vs_performance(self):
        xs = [0, 1, 3, 4, 6, 7, 8, 9]
        model_idxs = self.get_models_for_grid_size_comparison()
        rmse_ys = [self.rmse_values_dict[self.model_names[i]] for i in model_idxs]
        rmse_stds = [self.rmse_stds_dict[self.model_names[i]] for i in model_idxs]
        mae_ys = [self.mae_values_dict[self.model_names[i]] for i in model_idxs]
        mae_stds = [self.mae_stds_dict[self.model_names[i]] for i in model_idxs]
        miou_ys = [self.miou_values_dict[self.model_names[i]] for i in model_idxs]
        miou_stds = [self.miou_stds_dict[self.model_names[i]] for i in model_idxs]
        cs = [self.model_colour(self.model_names[i]) for i in model_idxs]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

        axes[0].bar(xs, rmse_ys, color=cs, yerr=rmse_stds)
        axes[0].set_xticks([0.5, 3.5, 7.5])
        axes[0].set_xticklabels(['8', '16', '32'])
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[0].set_ylabel('RMSE')

        axes[1].bar(xs, mae_ys, color=cs, yerr=mae_stds)
        axes[1].set_xticks([0.5, 3.5, 7.5])
        axes[1].set_xticklabels(['8', '16', '32'])
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[1].set_xlabel('Grid Size')
        axes[1].set_ylabel('MAE')

        axes[2].bar(xs, miou_ys, color=cs, yerr=miou_stds)
        axes[2].set_xticks([0.5, 3.5, 7.5])
        axes[2].set_xticklabels(['8', '16', '32'])
        axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[2].set_ylabel('mIoU')

        labels = ['SR', 'MRSO', 'MRMO s=\\{0,1,2\\}', 'MRMO s=m']
        legend_colours = [self.model_colour(self.model_names[i]) for i in [0, 9, 10, 13]]
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in legend_colours]
        fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    def plot_model_size_vs_performance(self):
        model_names = self.grid_sizes.keys()
        xs = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        rmse_ys = [self.rmse_values_dict[model] for model in model_names][:9]
        rmse_stds = [self.rmse_stds_dict[model] for model in model_names][:9]
        mae_ys = [self.mae_values_dict[model] for model in model_names][:9]
        mae_stds = [self.mae_stds_dict[model] for model in model_names][:9]
        miou_ys = [self.miou_values_dict[model] for model in model_names][:9]
        miou_stds = [self.miou_stds_dict[model] for model in model_names][:9]
        cs = self.colour_cycle[:3] * 3

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

        axes[0].bar(xs, rmse_ys, color=cs, yerr=rmse_stds)
        axes[0].set_xticks([1, 5, 9])
        axes[0].set_xticklabels(['8', '16', '32'])
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[0].set_ylabel('RMSE')

        axes[1].bar(xs, mae_ys, color=cs, yerr=mae_stds)
        axes[1].set_xticks([1, 5, 9])
        axes[1].set_xticklabels(['8', '16', '32'])
        axes[1].set_xlabel('Grid Size')
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[1].set_ylabel('MAE')

        axes[2].bar(xs, miou_ys, color=cs, yerr=miou_stds)
        axes[2].set_xticks([1, 5, 9])
        axes[2].set_xticklabels(['8', '16', '32'])
        axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        axes[2].set_ylabel('mIoU')

        labels = ['Small', 'Medium', 'Large']
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, linewidth=0) for c in cs]
        fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()


if __name__ == "__main__":
    run()
