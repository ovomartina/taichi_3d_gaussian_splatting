import numpy as np
import pandas as pd
import argparse
import json
from taichi_3d_gaussian_splatting.utils import inverse_SE3, SE3_to_quaternion_and_translation_torch
import torch
import os
import sym
import matplotlib.pyplot as plt

def quaternion_multiply_numpy(
    q0,
    q1,
):
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    return torch.stack([x, y, z, w], dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_groundtruth_path", type=str, required=True)
    parser.add_argument("--json_colmap_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.json_groundtruth_path, 'r') as json_gt:
        groundtruth_data = json.load(json_gt)
    with open(args.json_colmap_path, 'r') as json_colmap:
        colmap_data = json.load(json_colmap)

    errors_q = []
    errors_t = []
    
    for i, colmap_element in enumerate(colmap_data):
        groundtruth_element = groundtruth_data[i]
        groundtruth_colmap_camera = torch.tensor(
            groundtruth_element["T_colmap_camera"])
        colmap_camera = torch.tensor(colmap_element["T_pointcloud_camera"])
        q_gt, t_gt = SE3_to_quaternion_and_translation_torch(
            groundtruth_colmap_camera.unsqueeze(0))
        q_colmap, t_colmap = SE3_to_quaternion_and_translation_torch(
            colmap_camera.unsqueeze(0))

        # groundtruth_rotation = sym.Rot3(
        #     q_gt.detach().cpu().numpy().reshape((1, 4)))
        # groundtruth_pose = sym.Pose3(
        #     R=groundtruth_rotation, t=t_gt.detach().cpu().numpy().reshape((1, 3)).astype("float"))

        # colmap_rotation = sym.Rot3(
        #     q_colmap.detach().cpu().numpy().reshape((1, 4)))
        # colmap_pose = sym.Pose3(
        #     R=groundtruth_rotation, t=t_colmap.detach().cpu().numpy().reshape((1, 3)).astype("float"))

        # q_gt = q_gt.detach().cpu().numpy().reshape((1, 4))
        q_gt_inverse = q_gt * \
            np.array([-1., -1., -1., 1.])
        q_difference = quaternion_multiply_numpy(q_gt_inverse.reshape(
            (1, 4)), q_colmap.reshape((1, 4)),)
        q_difference = q_difference.cpu().numpy()
        angle_difference = np.abs(
            2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))
        if angle_difference > np.pi:
            angle_difference = 2*np.pi - angle_difference
        errors_q.append(angle_difference)
        
        error_t = torch.linalg.vector_norm(
                            t_colmap - t_gt)
        error_t = t_colmap - t_gt
        errors_t.append(error_t)

    errors_q_numpy = np.array(errors_q)
    errors_t_numpy = np.array(errors_t)
    errors_t_numpy = np.squeeze(errors_t_numpy,axis=1)
    print(errors_t_numpy.shape)
    print("mean q error: ", errors_q_numpy.mean())
    print("mean t error: ", errors_t_numpy.mean())
    plt.figure()
    plt.plot(errors_q_numpy)
    plt.xlabel("Item")
    plt.ylabel("Error")
    plt.title("Rotational error")
    plt.savefig(
        "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/colmap_rot_error.png")
    plt.clf()
    
    plt.plot(errors_t_numpy[ :, 0])
    plt.plot(errors_t_numpy[:, 1])
    plt.plot(errors_t_numpy[:, 2])
    plt.xlabel("Item")
    plt.ylabel("Error")
    plt.title("Translational error")
    plt.savefig(
        "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/colmap_trasl_error.png")
    plt.clf()