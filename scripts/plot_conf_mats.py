# import seaborn as sns
# import matplotlib.pyplot as plt

# Row normalise
# import torch.nn.functional as f
# norm_test_conf_mat = f.normalize(mean_test_conf_mat, p=1, dim=1)
# norm_test_conf_mat = torch.nan_to_num(norm_test_conf_mat, 0)
#
# fig, axis = plt.subplots(nrows=1, ncols=1)
# sns.heatmap(norm_test_conf_mat, ax=axis, fmt='', cbar=False, cmap='viridis',
#             vmin=0, vmax=torch.max(norm_test_conf_mat))
# axis.set_aspect('equal')
# axis.tick_params(axis='x', which='major', top=False, bottom=True, labeltop=False, labelbottom=True)
# axis.tick_params(axis='y', which='major', left=False, right=True, labelleft=False, labelright=True,
#                  labelrotation=0)
# axis.set_xlabel('Predicted Label')
# axis.yaxis.set_label_position("right")
# axis.set_ylabel('True Label')
# axis.set_facecolor("black")
# plt.show()

# per_class_recall = torch.diag(mean_test_conf_mat) / torch.sum(mean_test_conf_mat, dim=1)
