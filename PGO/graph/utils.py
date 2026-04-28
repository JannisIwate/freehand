from gtsam import Rot3, Point3, Pose3
import numpy as np


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