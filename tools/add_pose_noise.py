import argparse
import json
import os
import re
import numpy as np
import torch
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch, perturb_pose_quaternion_translation_torch, quaternion_to_rotation_matrix_torch
import sym


def add_pose_noise(json_path: str, noise_q: float, noise_t: float):
    
    with open(json_path, 'r') as json_file:
        groundtruth_data = json.load(json_file)
    for entry in groundtruth_data:
        T_pointcloud_camera = torch.tensor(entry["T_pointcloud_camera"])
        q_pointcloud_camera, t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            T_pointcloud_camera.unsqueeze(0))
        rotation_noise = np.random.normal(loc=0, scale=noise_q, size=3)
        translation_noise = np.random.normal(loc=0, scale=noise_t, size=3) #np.random.normal(loc=0, scale=0.0, size=3)
        noise_lie_std = np.concatenate((rotation_noise, translation_noise))
        
        # q_pointcloud_camera, t_pointcloud_camera = perturb_pose_quaternion_translation_torch(
        #     q_pointcloud_camera, t_pointcloud_camera, noise_q, noise_t)
        q_pointcloud_camera_numpy = q_pointcloud_camera.cpu().numpy()
        t_pointcloud_camera_numpy = t_pointcloud_camera.cpu().numpy()

        gt_rotation = sym.Rot3(
            q_pointcloud_camera_numpy.reshape((4, 1)))
        gt_pose = sym.Pose3(
            R=gt_rotation, t=t_pointcloud_camera_numpy.astype("float"))

        perturbed_pose = sym.Pose3.retract(
            gt_pose, noise_lie_std, 1e-8)

        perturbed_q_pointcloud_camera = torch.tensor(
            [perturbed_pose.rotation().data[:]]).to(torch.float32)
        perturbed_t_pointcloud_camera = torch.tensor(
            [perturbed_pose.position()]).to(torch.float32)
        # R = quaternion_to_rotation_matrix_torch(q_pointcloud_camera)
        # T_pointcloud_camera_perturbed = torch.vstack((torch.hstack((R.squeeze(0), t_pointcloud_camera.reshape((3, 1)))),\
        #                                 torch.tensor([0.,0.,0.,1.])))
        R = quaternion_to_rotation_matrix_torch(perturbed_q_pointcloud_camera)
        T_pointcloud_camera_perturbed = torch.vstack((torch.hstack((R.squeeze(0), perturbed_t_pointcloud_camera.reshape((3, 1)))),\
                                        torch.tensor([0.,0.,0.,1.])))
        entry["T_pointcloud_camera_perturbed"] = T_pointcloud_camera_perturbed.cpu().tolist()
    json_object = json.dumps(groundtruth_data, indent=4)
    print("output")
    print(os.path.join(os.path.dirname(json_path),
          os.path.splitext(os.path.basename(json_path))[0])+".json")
    # print(json_object)
    with open(os.path.join(os.path.dirname(json_path),os.path.splitext(os.path.basename(json_path))[0])+".json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train_path", type=str, required=True)
    parser.add_argument("--json_val_path", type=str, required=True)
    parser.add_argument("--json_test_path", type=str, required=False)
    parser.add_argument("--noise_q", type=float, required=True)
    parser.add_argument("--noise_t", type=float, required=True)
    args = parser.parse_args()

    add_pose_noise(args.json_train_path, args.noise_q, args.noise_t)
    add_pose_noise(args.json_val_path, args.noise_q, args.noise_t)
