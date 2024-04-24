
import torch
import numpy as np

import os
import PIL.Image
import torchvision
import argparse
import json
import taichi as ti
import sym
import matplotlib.pyplot as plt
import torchvision
import open3d as o3d
import pypose as pp
from pytorch_msssim import ssim
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as pl
import pickle

from dataclasses import dataclass
from typing import List, Tuple


from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudContinuousPoseRasterisation import GaussianPointCloudContinuousPoseRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer

import plotCoordinateFrame

# DEBUG - allow reproducibility
torch.manual_seed(42)
np.random.seed(42)

_EPS = 1e-6
M = (1/6)*torch.tensor([[5, 3, -3, 1],
                        [1, 3, 3, -2],
                        [0, 0, 0, 1]], device="cuda")

_NOISE_Q = 0.02
_NOISE_T = 0.02

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


def interpolate_bspline(time: float, bases: torch.Tensor) -> torch.Tensor:
    tt = torch.pow(time, torch.arange(0, 4, device="cuda"))
    w = torch.matmul(M, tt)

    delta_pose = bases[0] + w[0]*(bases[1, :] - bases[0, :]) + w[1]*(
        bases[2, :] - bases[1, :]) + w[2]*(bases[3, :]-bases[2, :])
    
    return delta_pose

def evaluate_spline_bases(poses: np.array): #poses: Nx6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(),tt)
        
        section = np.array([1-w[0], w[0]-w[1],w[1]-w[2], w[2]]).reshape((1,4)) # .repeat(6, axis=0)
        A[i,:]= section
    # poses = np.reshape(poses, (num_sample_poses*6,1))

    bases, _, _, _ = np.linalg.lstsq(A, poses)
    return np.array(bases).reshape(4,6)
    
def quaternion_difference_rad(ref_quaternion: torch.Tensor, current_quaternion: torch.Tensor):
    ref_quaternion = ref_quaternion.cuda()
    current_quaternion = current_quaternion.cuda()
    q_pointcloud_camera_gt_inverse = ref_quaternion * \
        torch.tensor([-1., -1., -1., 1.], device="cuda")
    q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
        (1, 4)), current_quaternion.reshape((1, 4)).cuda(),)
    q_difference = q_difference.cpu().numpy()
    angle_difference = np.abs(
        2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))
    if angle_difference > np.pi:
        angle_difference = 2*np.pi - angle_difference
    return angle_difference


class PoseEstimator():
    @dataclass
    class PoseEstimatorConfig:
        device: str = "cuda"
        image_height: int = 405
        image_width: int = 720
        camera_intrinsics: torch.Tensor = torch.tensor([[400, 0, 360],
                                                        [0, 400, 202.5],
                                                        [0, 0, 1]], device="cuda")
        parquet_path: str = None
        initial_guess_T_pointcloud_camera = torch.eye(4)
        image_path_list: List[str] = None
        json_file_path: str = None

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, output_path, config) -> None:
        self.config = config
        self.output_path = output_path
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16

        scene = GaussianPointCloudScene.from_parquet(
            self.config.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )

        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            self.config.image_width, self.config.image_height))

    def _merge_scenes(self, scene_list):
        merged_point_cloud = torch.cat(
            [scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = torch.cat(
            [scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = [scene.point_cloud.shape[0]
                              for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene_list[idx].point_cloud.mean(dim=0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            ))
        return merged_scene

    def start(self):
        d = self.config.image_path_list

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(os.path.join(self.output_path,"pickle_files")):
            os.makedirs(os.path.join(self.output_path,"pickle_files"))
        if not os.path.exists(os.path.join(self.output_path,"2d_plots")):
            os.makedirs(os.path.join(self.output_path,"2d_plots"))
        if not os.path.exists(os.path.join(self.output_path,"3d_plots")):
            os.makedirs(os.path.join(self.output_path,"3d_plots"))
            
        with open(self.config.json_file_path) as f:
            d = json.load(f)

            total_images = len(d)
            batch_size = 4  # Depends on how many measurements in time - straightness of trajectory
            indices = [0, int(batch_size*0.33), int(batch_size*0.66), -1]
            num_batches = total_images//batch_size
            if total_images % batch_size != 0:
                print("Discarding images")

            # Batch images
            for i in range(num_batches):
                if i > 0:
                    continue
                print(
                    f"=================Batch {i}========================")
                view = d[i*batch_size:(i+1)*batch_size]

                ground_truth_image_torch_list = []
                groundtruth_T_pointcloud_camera_torch_list = []
                perturbed_T_pointcloud_camera_torch_list = []

                for view_dict in view:
                    # Load groundtruth image path from the current dictionary
                    ground_truth_image_path = view_dict["image_path"]
                    print(f"Loading image {ground_truth_image_path}")

                    # Load the image and convert it to a numpy array
                    ground_truth_image_numpy = np.array(
                        PIL.Image.open(ground_truth_image_path))
                    ground_truth_image_tensor = torchvision.transforms.functional.to_tensor(
                        ground_truth_image_numpy)

                    # Append the image tensor to a list

                    ground_truth_images, resized_camera_info, resized_depth_image = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image_tensor,
                                                                                                                                                None,
                                                                                                                                                self.camera_info,
                                                                                                                                                1)
                    ground_truth_images = ground_truth_images.cuda()
                    ground_truth_image_torch_list.append(ground_truth_images)

                    groundtruth_T_pointcloud_camera = torch.tensor(
                        view_dict["T_pointcloud_camera"],
                        device="cuda").unsqueeze(0)
                    perturbed_T_pointcloud_camera = torch.tensor(
                        view_dict["T_pointcloud_camera_perturbed"],
                        device="cuda").unsqueeze(0)

                    groundtruth_T_pointcloud_camera_torch_list.append(
                        groundtruth_T_pointcloud_camera)
                    perturbed_T_pointcloud_camera_torch_list.append(
                        perturbed_T_pointcloud_camera)

                # Extract all in batch
                groundtruth_T_pointcloud_camera_torch = torch.stack(
                    groundtruth_T_pointcloud_camera_torch_list, dim=0)
                groundtruth_T_pointcloud_camera_torch = groundtruth_T_pointcloud_camera_torch.squeeze(
                    1)
                groundtruth_q_pointcloud_camera, groundtruth_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                    groundtruth_T_pointcloud_camera_torch)

                perturbed_T_pointcloud_camera_torch = torch.stack(
                    perturbed_T_pointcloud_camera_torch_list, dim=0)
                perturbed_T_pointcloud_camera_torch = perturbed_T_pointcloud_camera_torch.squeeze(
                    1)
                perturbed_q_pointcloud_camera, perturbed_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                    perturbed_T_pointcloud_camera_torch)

                # Convert perturbed poses into lie algebra represenation
                groundtruth_lie = []
                perturbed_lie = []
                groundtruth_pose = []
                perturbed_pose = []

                for j in range(batch_size):
                    rotation_groundtruth = sym.Rot3(
                        groundtruth_q_pointcloud_camera[j].cpu().numpy())
                    pose_groundtruth = sym.Pose3(
                        R=rotation_groundtruth, t=groundtruth_t_pointcloud_camera[j].cpu().numpy())
                    groundtruth_lie.append(pose_groundtruth.to_tangent(_EPS))
                    groundtruth_pose.append(pose_groundtruth)

                    perturbed_rotation = sym.Rot3(
                        perturbed_q_pointcloud_camera[j].cpu().numpy())
                    pose_perturbed = sym.Pose3(
                        R=perturbed_rotation, t=perturbed_t_pointcloud_camera[j].cpu().numpy())
                    perturbed_lie.append(pose_perturbed.to_tangent(_EPS))
                    perturbed_pose.append(pose_perturbed)

                # Compute groundtruth spline bases for the batch
                perturbed_bases = evaluate_spline_bases(np.array(
                    [perturbed_pose[k].to_tangent(_EPS) for k in range(batch_size)]
                ))
                
                groundtruth_bases = evaluate_spline_bases(np.array(
                    [groundtruth_pose[k].to_tangent(_EPS) for k in range(batch_size)]
                ))
                
                # # Perturb the spline bases as initialization
                # noise_q = np.random.normal(loc=0, scale=_NOISE_Q, size=(4,3))
                # noise_t = np.random.normal(loc=0, scale=_NOISE_T, size=(4,3))
                # noise = np.hstack([noise_q, noise_t])
                # bspline_bases_numpy = groundtruth_bases + noise
                
                # I could have also evaluated the initial spline on the noisy discrete poses. More correct
                bspline_bases = torch.tensor(perturbed_bases).reshape((4, 6)).cuda()  # 4 bases represented by Vec6
                bspline_bases.requires_grad_()


                self.rasteriser = GaussianPointCloudContinuousPoseRasterisation(
                    config=GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationConfig(
                        near_plane=0.001,
                        far_plane=1000.,
                        depth_to_sort_key_scale=100.,
                        enable_depth_grad=True,
                    ))

                # Optimization starts
                optimizer_bspline_bases = torch.optim.Adam(
                    [bspline_bases], lr=1e-3, betas=(0.9, 0.999))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_bspline_bases, gamma=0.9947)

                num_epochs = 3000

                errors_t = np.zeros((batch_size, num_epochs))
                errors_q = np.zeros((batch_size, num_epochs))

                # Pose refinement
                for epoch in range(num_epochs):
                    # Set the gradient to zero
                    optimizer_bspline_bases.zero_grad()

                    # Cumulate gradients from all images in segment
                    # L1 = 0

                    for i, view_dict in enumerate(view):
                        current_t = i / batch_size  # Since I assume measurements equally spaced in time
                        # Plot pose error
                        with torch.no_grad():
                            pose = interpolate_bspline(
                                current_t, bspline_bases)
                            pose = sym.Pose3.from_tangent(pose)
                            errors_t[i, epoch] = np.linalg.norm(
                                np.array(pose.position()) - np.array(groundtruth_pose[i].position()))
                            errors_q[i, epoch] = quaternion_difference_rad(torch.tensor(
                                [pose.rotation().data[:]]).to(torch.float32), torch.tensor(
                                [groundtruth_pose[i].rotation().data[:]]).to(torch.float32))
                            
                        predicted_image, predicted_depth, _, _ = self.rasteriser(
                            GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                point_object_id=self.scene.point_object_id,
                                bases=bspline_bases,
                                time=current_t,
                                camera_info=resized_camera_info,
                                color_max_sh_band=3,
                            )
                        )

                        predicted_image = torch.clamp(
                            predicted_image, min=0, max=1)
                        predicted_image = predicted_image.permute(2, 0, 1)

                        predicted_depth = predicted_depth.cuda()
                        predicted_depth = predicted_depth / \
                            torch.max(predicted_depth)
                        ground_truth_image = ground_truth_image_torch_list[i]
                        if len(predicted_image.shape) == 3:
                            predicted_image_temp = predicted_image.unsqueeze(0)
                        if len(ground_truth_image.shape) == 3:
                            ground_truth_image_temp = ground_truth_image.unsqueeze(
                                0)

                        # sum over all images
                        L1 = 0.8*torch.abs(predicted_image - ground_truth_image).mean() + 0.2*(1 - ssim(predicted_image_temp, ground_truth_image_temp,
                                                                                                        data_range=1, size_average=True))
                        L = L1
                        L.backward()

                        if (not torch.isnan(bspline_bases.grad).any()):
                            optimizer_bspline_bases.step()

                    if epoch % 20 == 0:
                        scheduler.step()
                        for param_group in optimizer_bspline_bases.param_groups:
                            if param_group['lr'] < 1e-5:
                                param_group['lr'] = 1e-5

                    if epoch % 100 == 0:
                        print(L)
                        # DEBUG visualization ===========================
                        temp_data = []

                        # Iterate over groundtruth
                        for i, pose in enumerate(groundtruth_pose):
                            position_data = np.array(
                                pose.position().data[:]).reshape(1, 3)
                            rotation_data = np.array(
                                pose.rotation().data[:]).reshape(1, 4)
                            temp_data.append(
                                np.hstack([position_data, rotation_data]))

                        temp_array = np.array(temp_data)
                        temp_reshaped = temp_array.reshape((batch_size, 7))
                        temp_reshaped = temp_reshaped[[
                            indices[0], indices[1], indices[2], indices[3]], :]

                        # First position, then quaternion                       
                        # poses = pp.SE3(temp_reshaped)
                        poses = pp.se3(groundtruth_bases.reshape(4,6))
                        poses = pp.Exp(poses)
                        f1 = pl.figure(1)
                        a3d = f1.add_subplot(111, projection='3d')

                        f2 = pl.figure(2)
                        a2d = f2.add_subplot(111)

                        a3d.set_xlabel('X')
                        a3d.set_ylabel('Y')
                        a3d.set_zlabel('Z')

                        # Plot current trajectory guess
                        with torch.no_grad():
                            bspline_bases_numpy = bspline_bases.clone().detach().cpu().numpy()
                            plotCoordinateFrame.plot_trajectory(
                                a3d, bspline_bases_numpy[:, 3:], color="black", linewidth=1, resolution=0.1, label="estimation")
                            plotCoordinateFrame.plot_trajectory_2d(
                                a2d, bspline_bases_numpy[:, 3:], color="black", linewidth=1, resolution=0.1, label="estimation")
                            
                            # Add scatter for reconstructed points
                            current_t = i / batch_size
                            # Plot pose error
                            t_step = 1/batch_size
                            t_range = t_step*np.arange(1, batch_size-1)
                            poses = np.array([interpolate_bspline(
                                t, bspline_bases).clone().detach().cpu().numpy() for t in t_range])
                            a2d.scatter(poses[:,3],poses[:,4], s=5, color="black")
                            a3d.scatter(poses[:,3],poses[:,4], poses[:,5], s=5, color="black")
                            
                            # Add scatter for reconstructed bases
                            a2d.scatter(bspline_bases_numpy[:,3],bspline_bases_numpy[:,4], s=5, color="gray")
                            a3d.scatter(bspline_bases_numpy[:,3],bspline_bases_numpy[:,4], bspline_bases_numpy[:,5], s=5, color="gray")
                        
                        # Plot groundtruth trajectory
                        plotCoordinateFrame.plot_trajectory_lie(
                            a3d, groundtruth_bases, linewidth=1, resolution=0.1, size=0.2)
                        plotCoordinateFrame.plot_trajectory(
                            a3d, groundtruth_bases[:,3:], color="orange", linewidth=1, label="groundtruth")
                        plotCoordinateFrame.plot_trajectory_2d(
                            a2d, groundtruth_bases[:,3:], color="orange", linewidth=1, label="groundtruth")
                        
                        groundtruth_delta = np.array([groundtruth_pose[k].to_tangent(_EPS) for k in range(batch_size)])
                        
                        # Add scatter for groundtruth points
                        a2d.scatter(groundtruth_delta[:,3],groundtruth_delta[:,4], s=5, color="orange")
                        a3d.scatter(groundtruth_delta[:,3],groundtruth_delta[:,4], groundtruth_delta[:,5], s=5, color="orange", label="Groundtruth discrete poses")

                        # Add scatter for groundtruth bases
                        a2d.scatter(groundtruth_bases[:,3],groundtruth_bases[:,4], s=5, color="red")
                        a3d.scatter(groundtruth_bases[:,3],groundtruth_bases[:,4], groundtruth_bases[:,5], s=5, color="red", label="Groundtruth spline bases")
                        
                        a3d.legend()
                        a2d.legend()

                        f1.savefig(os.path.join(
                            self.output_path,"3d_plots", f'figure_{epoch}.png'))
                        f2.savefig(os.path.join(
                            self.output_path,"2d_plots", f'figure_{epoch}_2d.png'))

                        pickle.dump(f1, open(os.path.join(
                            self.output_path, "pickle_files", f'FigureObject.fig_{epoch}.pickle'), 'wb'))
                        f1.clear()
                        f2.clear()

                        for i in range(batch_size):
                            plt.plot(errors_t[i, :epoch], label=f"Error frame {i}")
                            plt.xlabel("Epoch")
                            plt.ylabel("Error")
                            plt.title("Translational error")
                            plt.legend()
                        plt.savefig(
                            os.path.join(self.output_path, f"trasl_error.png"))
                        plt.clf()
                            
                        for i in range(batch_size):
                            plt.plot(errors_q[i, :epoch], label=f"Error frame {i}")
                            plt.xlabel("Epoch")
                            plt.ylabel("Error")
                            plt.title("Rotational error")
                            plt.legend()
                        plt.savefig(
                            os.path.join(self.output_path, f"rot_error_frame.png"))
                        plt.clf()
                        # Add mean?
                        # ============================================


"""
Main
"""
parser = argparse.ArgumentParser(description='Parquet file path')
parser.add_argument('--parquet_path', type=str, help='Parquet file path')
parser.add_argument('--json_file_path', type=str,
                    help='Json trajectory file path')
parser.add_argument('--output_path', type=str, help='Output folder path')

args = parser.parse_args()

print("Opening parquet file ", args.parquet_path)
parquet_path_list = args.parquet_path
ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
visualizer = PoseEstimator(args.output_path, config=PoseEstimator.PoseEstimatorConfig(
    parquet_path=parquet_path_list,
    json_file_path=args.json_file_path
))
visualizer.start()
