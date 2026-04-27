import os
from pathlib import Path
import torch
import numpy as np
import gtsam
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from freehand.utils import *
from gtsam import NonlinearFactorGraph, Values, noiseModel

#TODO build graph from GT and compare

# load data
BASE_PATH = os.path.join(os.getcwd(), "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

abs_poses_estimated = torch.load(BASE_PATH + '/pose_data/predictions.pt')            # (N, 3, 4), element (frame), row (coordinate), column (corner)
rel_poses_estimated = torch.load(BASE_PATH + '/pose_data/predictions_transforms_local.pt') # (N, 4, 4)

abs_poses_GT = torch.load(BASE_PATH + '/pose_data/labels.pt') # (N, 3, 4)
rel_poses_GT = torch.load(BASE_PATH + '/pose_data/predictions_transforms_gt.pt') # (N, 4, 4)

# remove initial zero element
abs_poses_estimated = abs_poses_estimated[1:]
rel_poses_estimated = rel_poses_estimated[1:-1] # remove last element as it is not needed

def extract_positions(values):
    xs, ys, zs = [], [], []
    for i in range(values.size()):
        p = values.atPose3(i).translation()

        # works for both numpy and Point3
        if isinstance(p, np.ndarray):
            x, y, z = p
        else:
            x, y, z = p.x(), p.y(), p.z()

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs)

def build_graph(abs_poses, rel_poses, optimize = True):
    N = abs_poses.shape[0]

    # build graph
    graph = NonlinearFactorGraph()
    initial = Values()

    # gaussian noise models (tune these!)
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-6]*6)) # small value to anchor first pose (equals in large penalty regarding error function, first pose shall be fixed)
    odom_noise  = noiseModel.Diagonal.Sigmas(np.array([0.05]*6)) # larger value vice versa

    # Insert nodes (absolute poses)
    for i in range(N):
        pose = mat4_to_pose3(abs_poses[i])
        initial.insert(i, pose)

    # anchor first pose, prior
    graph.add(
        gtsam.PriorFactorPose3(
            0,
            initial.atPose3(0),
            prior_noise
        )
    )

    # add odometry edges
    for i in range(N-1): # one less edge than nodes (without prior)
        rel_pose = mat34_to_pose3(rel_poses[i])

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                rel_pose,
                odom_noise
            )
        )

    print(f"Graph: {graph.size()} factors, {N} poses")

    # optimize
    optimized = None
    if optimize:
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        optimized = optimizer.optimize()

    return initial, optimized

def plot_trajectories(trajectories, labels=None, colors=None):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = len(trajectories)

    # defaults
    if labels is None:
        labels = [f"traj_{i}" for i in range(n)]
    if colors is None:
        colors = [None] * n

    for i, (xs, ys, zs) in enumerate(trajectories):
        ax.plot(xs, ys, zs,
                label=labels[i],
                color=colors[i])

    ax.scatter(xs[0], ys[0], zs[0], color=colors[i])

    ax.set_title("Pose Graph Trajectories")
    ax.legend()
    plt.show()

initial_estimated, optimized_estimated = build_graph(abs_poses_estimated, rel_poses_estimated, True)
initial_GT, _ = build_graph(abs_poses_GT, rel_poses_GT, False)
print(initial_estimated.__sizeof__())
print(initial_GT.__sizeof__())

plot_trajectories([
    extract_positions(initial_estimated),
    extract_positions(optimized_estimated),
    extract_positions(initial_GT)
    ],
    labels=["Initial estimated", "Optimized estimated", "GT"],
    colors=["blue", "cyan", "red"])