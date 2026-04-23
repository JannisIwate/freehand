import os
from pathlib import Path
import torch
import numpy as np
import gtsam
import sys
sys.path.append(os.getcwd())
from freehand.utils import *
from gtsam import NonlinearFactorGraph, Values, noiseModel, Rot3, Point3, Pose3

# -----------------------------
# Load data
# -----------------------------
BASE_PATH = os.path.join(os.getcwd(), "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

abs_poses_torch = torch.load(BASE_PATH + '/pose_data/predictions.pt')            # (N, 3, 4), element (frame), row (coordinate), column (corner)
rel_poses_torch = torch.load(BASE_PATH + '/pose_data/predictions_transforms_local.pt') # (N, 4, 4)

N = abs_poses_torch.shape[0]

def pose3_to_mat4(pose):
    R = pose.rotation().matrix()
    t = pose.translation()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x(), t.y()]

    return T


def mat4_to_pose3(T):
    T = T.cpu().numpy()

    R = Rot3(T[:3, :3])
    t = Point3(*T[:3, 3])

    return Pose3(R, t)


def mat34_to_pose3(T):
    """Convert 3x4 -> Pose3"""
    T = T.cpu().numpy()

    R = Rot3(T[:3, :3])
    t = Point3(*T[:3, 3])

    return Pose3(R, t)

# -----------------------------
# Build graph
# -----------------------------
graph = NonlinearFactorGraph()
initial = Values()

# Noise models (tune these!)
prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
odom_noise  = noiseModel.Diagonal.Sigmas(np.array([0.05]*6))

# -----------------------------
# Insert nodes (absolute poses)
# -----------------------------
for i in range(N-1):
    pose = mat4_to_pose3(abs_poses_torch[i+1])
    initial.insert(i, pose)

# -----------------------------
# Anchor first pose
# -----------------------------
graph.add(
    gtsam.PriorFactorPose3(
        0,
        initial.atPose3(0),
        prior_noise
    )
)

# -----------------------------
# Add odometry edges
# -----------------------------
# IMPORTANT:
# rel_poses_torch[i] should represent T_{i -> i+1}
# So we connect i -> i+1

for i in range(N - 1):
    rel_pose = mat34_to_pose3(rel_poses_torch[i+1])

    graph.add(
        gtsam.BetweenFactorPose3(
            i,
            i + 1,
            rel_pose,
            odom_noise
        )
    )

print(f"Graph: {graph.size()} factors, {N-1} poses")

# -----------------------------
# Optimize (optional)
# -----------------------------
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

# -----------------------------
# Extract trajectory
# -----------------------------
# def extract_positions(values):
#     xs, ys, zs = [], [], []
#     for i in range(values.size()-1):
#         p = values.atPose3(i).translation()
#         xs.append(p.x())
#         ys.append(p.y())
#         zs.append(p.z())
#     return np.array(xs), np.array(ys), np.array(zs)

# xs_i, ys_i, zs_i = extract_positions(initial)
# xs_r, ys_r, zs_r = extract_positions(result)

# # -----------------------------
# # Visualization
# # -----------------------------
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.plot(xs_i, ys_i, zs_i, label="Initial", linestyle="--")
# ax.plot(xs_r, ys_r, zs_r, label="Optimized")

# ax.scatter(xs_i[0], ys_i[0], zs_i[0], label="Start")
# ax.legend()

# plt.title("Pose Graph Trajectory")
# plt.show()