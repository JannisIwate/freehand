import torch
import os
import sys
sys.path.append(os.getcwd())
from graph.build_graph import *

# TODO try on all samples

# load data
BASE_PATH = os.path.join(os.getcwd(), "../freehand_adapted", "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

abs_poses_estimated = torch.load(BASE_PATH + '/pose_data/predictions.pt') # (N, 3, 4), element (frame), row (coordinate), column (corner)
rel_poses_estimated = torch.load(BASE_PATH + '/pose_data/predictions_transforms_local.pt') # (N, 4, 4)

abs_poses_GT = torch.load(BASE_PATH + '/pose_data/labels.pt') # (N, 3, 4)
rel_poses_GT = torch.load(BASE_PATH + '/pose_data/predictions_transforms_gt.pt') # (N, 4, 4)

# remove initial zero element
abs_poses_estimated = abs_poses_estimated[1:]
rel_poses_estimated = rel_poses_estimated[1:-1] # remove last element as it is not needed

# build graphs
initial_estimated, optimized_estimated = build_graph(abs_poses_estimated, rel_poses_estimated, True)
initial_GT, _ = build_graph(abs_poses_GT, rel_poses_GT, False)

# plot
plot_trajectories([
    extract_positions(initial_estimated),
    extract_positions(optimized_estimated),
    extract_positions(initial_GT)
    ],
    labels=["Initial estimated", "Optimized estimated", "GT"],
    colors=["blue", "cyan", "red"])