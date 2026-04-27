import numpy as np
import gtsam
import matplotlib.pyplot as plt
from gtsam import NonlinearFactorGraph, Values, noiseModel
from graph.utils import *


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