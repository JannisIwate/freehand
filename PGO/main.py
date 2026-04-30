import torch
import os
import sys
sys.path.append(os.getcwd())
from graph.build_graph import *
from graph.error_metrics import *


# TODO check why gt error is so huge

# load data
BASE_PATH = os.path.join(os.getcwd(), "../freehand_adapted", "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

inbetween_transforms_pred = torch.load(BASE_PATH + '/pose_data/inbetween_transforms_pred.pt') # (N, 4, 4)
acc_transforms_pred = torch.load(BASE_PATH + '/pose_data/acc_transforms_pred.pt') # (N, 4, 4)

inbetween_transforms_gt = torch.load(BASE_PATH + '/pose_data/inbetween_transforms_gt.pt') # (N, 4, 4)
acc_transforms_gt = torch.load(BASE_PATH + '/pose_data/acc_transforms_gt.pt') # (N, 4, 4)

# remove initial zero element and last element
inbetween_transforms_pred = inbetween_transforms_pred[1:]
acc_transforms_pred = acc_transforms_pred[1:-1]
inbetween_transforms_gt = inbetween_transforms_gt[1:]
acc_transforms_gt = acc_transforms_gt[1:-1]

# build graphs
graph_estimated, initial_estimated, optimized_estimated = build_graph(inbetween_transforms_pred, acc_transforms_pred, True)
graph_gt, initial_gt, _ = build_graph(inbetween_transforms_gt, acc_transforms_gt, False)

# plot
# plot_trajectories([
#     extract_positions(acc_transforms_pred),
#     extract_positions(acc_transforms_gt)
#     ],
#     labels=["Initial estimated", "GT"],
#     colors=["blue", "red"])

# error metrics
ge_error = graph_estimated.error(initial_estimated)
ge_error = graph_estimated.error(optimized_estimated)
ggt_error = graph_gt.error(initial_gt)

print(f"Initial error ge: {ge_error}")
print(f"Optimized error ge: {ge_error}")
print(f"Initial error ggt: {ggt_error}")

avg_t_err, avg_r_err = avg_trajectory_error(acc_transforms_gt, acc_transforms_pred)
print(f"Average translation error acc: {avg_t_err}")
print(f"Average rotation error acc\n: {avg_r_err}")

avg_t_err, avg_r_err = avg_trajectory_error(inbetween_transforms_gt, inbetween_transforms_pred)
print(f"Average translation error inbetween: {avg_t_err}")
print(f"Average rotation error inbetween\n: {avg_r_err}")