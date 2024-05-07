import os
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import curve_evaluation
import json
import pypose as pp
import sym
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch
import continuous_trajectory

folder = "continuous_trajectory_output_complete"
dirs = f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}"
groundtruth_trajectory_path = "scripts/data/val_path.obj"
json_file_path = "data/replica_colmap/room_1_high_quality_50_frames_test/val.json"

with open(groundtruth_trajectory_path, 'r') as obj_file:
    lines = obj_file.readlines()
vertices = []
for line in lines:
    if line.startswith('v '):
        vertex = line.split()[1:]
        vertex = [float(coord) for coord in vertex]
        vertices.append(vertex)
vertices_array = np.array(vertices)

files = os.listdir(dirs)
pt_files = [file for file in files if file.endswith('.pt')]

# Define a sorting key function to extract the number from the file name


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group())


# Sort the .pt files based on the extracted numbers
sorted_pt_files = sorted(pt_files, key=extract_number)

print(json_file_path)
with open(json_file_path) as f:
    d = json.load(f)
    N = 48
    batch_size = 4
    num_segments = 12
    index_in_segment = np.array([i % batch_size for i in range(N)])
    index_of_segment = np.array(
        range(num_segments)).repeat(batch_size, 0)
    time_index = index_in_segment*(1/batch_size)

    errors_q = np.zeros((N, len(sorted_pt_files)))
    errors_t = np.zeros((N, len(sorted_pt_files)))
    
    N = 35
    for current_file_number, file in enumerate(sorted_pt_files):
        file_path = os.path.join(dirs, file)
        bases = torch.load(file_path)

        pypose_bspline_knots = torch.zeros((bases.shape[0], 7))
        for base_number, base in enumerate(bases):
            base_lie = sym.Pose3.from_tangent(base)
            pypose_bspline_knots[base_number, :] = torch.hstack((torch.tensor([
                base_lie.position()]).to(torch.float32), torch.tensor([
                    base_lie.rotation().data[:]]).to(torch.float32)))

        # For all frames, compute position on b-spline and error w.r.t. groundtruth pose
        for n in range(N):
            initial_segment_index = index_of_segment[n]

            current_pose_tensor = curve_evaluation.cubic_bspline_interpolation(
                pp.SE3((pypose_bspline_knots[initial_segment_index:initial_segment_index +
                      4, :]).double()), u=torch.tensor([time_index[n]]).double(),
                enable_z_spline=True)
            current_pose = current_pose_tensor.clone().detach().cpu().numpy()
            
            # load groundtruth
            groundtruth_T_pointcloud_camera = torch.tensor(
                d[n]["T_pointcloud_camera"],
                device="cuda").unsqueeze(0)
            # groundtruth_T_pointcloud_camera_torch = torch.stack(
            #     groundtruth_T_pointcloud_camera, dim=0).squeeze(
            #     1)
            groundtruth_q_pointcloud_camera, groundtruth_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                groundtruth_T_pointcloud_camera)
            
            errors_t[n, current_file_number] = np.linalg.norm(
                np.array(current_pose[0, :3]) - np.array(groundtruth_t_pointcloud_camera.cpu().numpy()).reshape((1, 3)))
            errors_q[n, current_file_number] = continuous_trajectory.quaternion_difference_rad(torch.tensor(
                current_pose[0, 3:]), torch.tensor(groundtruth_q_pointcloud_camera).reshape((1, 4)))
            
    for n in range(N):
        plt.plot(errors_q[n, :])
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Rotational error")
        plt.legend()
        plt.savefig(
            os.path.join(dirs, f"final_rot_error_frame_{n}.png"))
        plt.clf()
        
        plt.plot(errors_t[n, :])
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Translational error")
        plt.legend()
        plt.savefig(
            os.path.join(dirs, f"final_trans_error_frame_{n}.png"))
        plt.clf()
    average_t_error = np.mean(errors_t, axis = 0)
    average_q_error = np.mean(errors_q, axis = 0)
    
    plt.plot(average_q_error[:])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Mean Rotational Error")
    plt.legend()
    plt.savefig(
        os.path.join(dirs, f"avg_rot_error_frame_{n}.png"))
    plt.clf()
    
    plt.plot(average_t_error[:])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Mean Translational Error")
    plt.legend()
    plt.savefig(
        os.path.join(dirs, f"avg_trans_error_frame_{n}.png"))
    plt.clf()