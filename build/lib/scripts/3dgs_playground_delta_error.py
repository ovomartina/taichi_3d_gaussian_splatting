import argparse
import json
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudPoseRasterisation import GaussianPointCloudPoseRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch
from taichi_3d_gaussian_splatting.Lidar import Lidar
from pytorch_msssim import ssim
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
import pylab as pl
import pickle

# %%
import os
import PIL.Image
import torchvision
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
import matplotlib.pyplot as plt
import torchvision
import sym
import open3d as o3d
import plotCoordinateFrame
import pypose as pp

# DEBUG - allow reproducibility
torch.manual_seed(42)
np.random.seed(42)

R_pontcloud_pointcloudstar = np.array([[-0.5614035,  0.7017544, -0.4385965],
                                       [-0.7017544, -0.1228070,  0.7017544],
                                       [0.4385965,  0.7017544,  0.5614035]])
t_pointcloud_pointcloudstar = np.array([-0.018, 0.006, 0.007]).T


def extract_q_t_from_pose(pose3: sym.Pose3) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.tensor(
        [pose3.rotation().data[:]]).to(torch.float32)
    t = torch.tensor(
        [pose3.position()]).to(torch.float32)
    return q, t


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
        initial_guess_T_pointcloud_camera = torch.tensor([
            [0.059764470905065536, 0.4444755017757416, -0.8937951326370239, 0.],
            [0.9982125163078308, -0.026611410081386566, 0.05351284518837929, 0.],
            [0.0, -0.8953956365585327, -0.44527140259742737, 0.],
            [0.0, 0.0, 0.0, 1.0]])
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

        # Initial guess

        self.initial_guess_T_pointcloud_camera = self.config.initial_guess_T_pointcloud_camera.to(
            self.config.device)
        initial_guess_T_pointcloud_camera = self.initial_guess_T_pointcloud_camera.unsqueeze(
            0)
        self.initial_guess_q_pointcloud_camera, self.initial_guess_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            initial_guess_T_pointcloud_camera)

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
        # the config does not matter here, only for training

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

        with open(self.config.json_file_path) as f:
            d = json.load(f)
            errors_t = []
            errors_q = []
            optimized_poses_lie = np.zeros((len(d), 6))
            groundtruth_poses_lie = np.zeros((len(d), 6))

            for count, view in enumerate(d):
                print(
                    f"=================Image {count}========================")
                if count <46:
                    continue

                # Load groundtruth image
                ground_truth_image_path = view["image_path"]
                print(f"Loading image {ground_truth_image_path}")
                ground_truth_image_numpy = np.array(
                    PIL.Image.open(ground_truth_image_path))
                ground_truth_image = torchvision.transforms.functional.to_tensor(
                    ground_truth_image_numpy)

                # DEBUG =============================================
                depth_image_path = view["depth_path"]
                print(f"Loading image {depth_image_path}")
                depth_image_path_numpy = np.array(
                    PIL.Image.open(depth_image_path))
                depth_image = torchvision.transforms.functional.to_tensor(
                    depth_image_path_numpy)
                # ===============================================

                ground_truth_image, resized_camera_info, resized_depth_image = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image,
                                                                                                                                           depth_image,
                                                                                                                                           self.camera_info,
                                                                                                                                           1)
                ground_truth_image = ground_truth_image.cuda()
                resized_depth_image = resized_depth_image.cuda()

                self.camera_info = resized_camera_info
                groundtruth_T_pointcloud_camera = torch.tensor(
                    view["T_pointcloud_camera"],
                    device="cuda")

                groundtruth_T_pointcloud_camera = groundtruth_T_pointcloud_camera.unsqueeze(
                    0)
                groundtruth_q_pointcloud_camera, groundtruth_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                    groundtruth_T_pointcloud_camera)
                print(
                    f"Ground truth q: \n\t {groundtruth_q_pointcloud_camera}")
                print(
                    f"Ground truth t: \n\t {groundtruth_t_pointcloud_camera}")

                print(
                    f"Ground truth transformation world to camera, in camera frame: \n\t {groundtruth_T_pointcloud_camera}")
                groundtruth_q_pointcloud_camera_numpy = groundtruth_q_pointcloud_camera.cpu().numpy()
                groundtruth_t_pointcloud_camera_numpy = groundtruth_t_pointcloud_camera.cpu().numpy()

                
                # Save groundtruth image
                im = PIL.Image.fromarray(
                    (ground_truth_image_numpy).astype(np.uint8))
                if not os.path.exists(os.path.join(self.output_path, f'groundtruth/')):
                    os.makedirs(os.path.join(self.output_path, 'groundtruth/'))
                im.save(os.path.join(self.output_path,
                        f'groundtruth/groundtruth_{count}.png'))

                # Get lidar file if available
                # DEBUG
                try:
                    lidar_path = view['lidar_path']
                    if lidar_path:
                        lidar_pcd = o3d.io.read_point_cloud(
                            view['lidar_path'])
                        lidar_pcd = torch.tensor(lidar_pcd.points)
                        T_lidar_camera = torch.tensor(
                            view['T_camera_lidar'])

                        if len(lidar_pcd) <= 0:
                            lidar_pcd = None
                            T_lidar_camera = None

                    lidar_measurement = Lidar(
                        lidar_pcd.cuda(), T_lidar_camera.cuda())
                except:
                    pass

                errors_t_current_pic = []
                errors_q_current_pic = []

                rotation_groundtruth = sym.Rot3(
                    groundtruth_q_pointcloud_camera_numpy[0, :])
                pose_groundtruth = sym.Pose3(
                    R=rotation_groundtruth, t=groundtruth_t_pointcloud_camera_numpy.T.astype("float"))

                groundtruth_poses_lie[count, :] = pose_groundtruth.to_tangent()

                delta_numpy_array_q = np.random.normal(0, 0.05, (3, 1))
                delta_numpy_array_t = np.random.normal(0, 0.05, (3, 1))
                delta_numpy_array = np.vstack(
                    (delta_numpy_array_q, delta_numpy_array_t))
                # delta_tensor = torch.zeros(
                #     (6, 1), requires_grad=True, device="cuda")
                delta_tensor_q = torch.zeros(
                    (3, 1), requires_grad=True, device="cuda")
                delta_tensor_t = torch.zeros(
                    (3, 1), requires_grad=True, device="cuda")
                print(f"Delta array:\n\t{delta_numpy_array}")

                epsilon = 0.0001
                initial_pose = sym.Pose3.retract(
                    pose_groundtruth, delta_numpy_array, epsilon)

                initial_q, initial_t = extract_q_t_from_pose(initial_pose)
                initial_q_numpy = initial_q.detach().cpu().numpy()
                initial_t_numpy = initial_t.detach().cpu().numpy()
                print(f"Initial pose:\n\t{initial_pose}")

                self.rasteriser = GaussianPointCloudPoseRasterisation(
                    config=GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationConfig(
                        near_plane=0.001,
                        far_plane=1000.,
                        depth_to_sort_key_scale=100.,
                        enable_depth_grad=True,
                    ))

                # Optimization starts
                optimizer_delta_q = torch.optim.Adam(
                    [delta_tensor_q], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)  #
                optimizer_delta_t = torch.optim.Adam(
                    [delta_tensor_t], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)  #
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_delta_q, gamma=0.9947)
                scheduler_t = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_delta_t, gamma=0.9947)
                # First: Coarse iterations
                num_coarse_epochs = 0  # 3000
                num_epochs = 1000  # 1000
                downsample_factor = 2
                ground_truth_image_downsampled, resized_camera_info_downsampled, _ = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image,
                                                                                                                                                 None,
                                                                                                                                                 self.camera_info,
                                                                                                                                                 downsample_factor)

                # Save coarse groundtruth image
                ground_truth_image_downsampled_numpy = ground_truth_image_downsampled.clone(
                ).detach().cpu().numpy()
                im = PIL.Image.fromarray(
                    (ground_truth_image_downsampled_numpy.transpose(1, 2, 0)*255).astype(np.uint8))
                im.save(os.path.join(self.output_path,
                        f'groundtruth/groundtruth_{count}_coarse.png'))

                for epoch in range(num_coarse_epochs):
                    # Set the gradient to zero
                    # optimizer_delta_coarse.zero_grad()
                    optimizer_delta_q.zero_grad()
                    optimizer_delta_t.zero_grad()

                    delta_tensor = torch.cat(
                        (delta_tensor_q, delta_tensor_t), axis=0)
                    # Compute current error
                    with torch.no_grad():
                        epsilon = 0.0001
                        delta_numpy_array = delta_tensor.clone().detach().cpu().numpy()
                        current_pose = initial_pose.retract(
                            delta_numpy_array, epsilon=epsilon)
                        current_q, current_t = extract_q_t_from_pose(
                            current_pose)
                        current_q_numpy_array = current_q.clone().detach().cpu().numpy()
                        current_t_numpy_array = current_t.clone().detach().cpu().numpy()
                        errors_t_current_pic.append(
                            current_t_numpy_array - groundtruth_t_pointcloud_camera.cpu().numpy())
                        # errors_q_current_pic.append(
                        #     current_q_numpy_array - groundtruth_q_pointcloud_camera.cpu().numpy())

                        # Quaternion error as rad
                        q_pointcloud_camera_gt_inverse = groundtruth_q_pointcloud_camera * \
                            torch.tensor([-1., -1., -1., 1.], device="cuda")
                        q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
                            (1, 4)), current_q.reshape((1, 4)).cuda(),)
                        q_difference = q_difference.cpu().numpy()
                        angle_difference = np.abs(
                            2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))
                        if angle_difference > np.pi:
                            angle_difference = 2*np.pi - angle_difference
                        errors_q_current_pic.append(angle_difference)

                    predicted_image, _, _, _ = self.rasteriser(
                        GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            delta=delta_tensor,
                            # initial_q=current_q_numpy_array,
                            # initial_t=current_t_numpy_array,
                            initial_q=initial_q_numpy,
                            initial_t=initial_t_numpy,
                            camera_info=resized_camera_info_downsampled,
                            color_max_sh_band=3,
                        )
                    )

                    predicted_image = torch.clamp(
                        predicted_image, min=0, max=1)
                    predicted_image = predicted_image.permute(2, 0, 1)

                    if len(predicted_image.shape) == 3:
                        predicted_image_temp = predicted_image.unsqueeze(0)
                    if len(ground_truth_image_downsampled.shape) == 3:
                        ground_truth_image_temp = ground_truth_image_downsampled.unsqueeze(
                            0)
                    L1 = 0.8*torch.abs(predicted_image - ground_truth_image_downsampled).mean() + 0.2*(1 - ssim(predicted_image_temp, ground_truth_image_temp,
                                                                                                                data_range=1, size_average=True))
                    L1.backward()

                    if (not torch.isnan(delta_tensor_q.grad).any()) and (not torch.isnan(delta_tensor_t.grad).any()):
                        optimizer_delta_q.step()
                        optimizer_delta_t.step()

                    if epoch % 5 == 0:
                        scheduler.step()
                        for param_group in optimizer_delta_q.param_groups:
                            if param_group['lr'] < 1e-5:
                                param_group['lr'] = 1e-5
                        scheduler_t.step()
                        for param_group in optimizer_delta_t.param_groups:
                            if param_group['lr'] < 1e-5:
                                param_group['lr'] = 1e-5

                    if epoch % 50 == 0:
                        print(L1)

                    if epoch % 300 == 0:
                        image_np = predicted_image.clone().detach().cpu().numpy()
                        im = PIL.Image.fromarray(
                            (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                        if not os.path.exists(os.path.join(self.output_path, f'epochs_delta_{count}/')):
                            os.makedirs(os.path.join(
                                self.output_path, f'epochs_delta_{count}/'))
                        im.save(os.path.join(self.output_path,
                                             f'epochs_delta_{count}/coarse_{epoch}.png'))

                epsilon = 0.0001
                delta_tensor = torch.cat(
                    (delta_tensor_q, delta_tensor_t), axis=0)
                delta_numpy_array = delta_tensor.clone().detach().cpu().numpy()
                current_pose = initial_pose.retract(
                    delta_numpy_array, epsilon=epsilon)
                print(f"Initial pose:\n\t{initial_pose}")
                print(
                    f"Current pose (after coarse refinement):\n\t{current_pose}")
                print(f"Ground truth pose:\n\t{pose_groundtruth}")

                # Pose refinement
                for epoch in range(num_epochs):
                    # Add error to plot
                    with torch.no_grad():
                        epsilon = 0.0001
                        delta_numpy_array = delta_tensor.clone().detach().cpu().numpy()
                        current_pose = initial_pose.retract(
                            delta_numpy_array, epsilon=epsilon)
                        T_pointcloud_camera_current = current_pose.to_homogenous_matrix()
                        current_q, current_t = extract_q_t_from_pose(
                            current_pose)
                        current_t_numpy_array = current_t.clone().detach().cpu().numpy()
                        errors_t_current_pic.append(
                            current_t_numpy_array - groundtruth_t_pointcloud_camera.cpu().numpy())  # Mantain axes

                        # Quaternion error as rad
                        q_pointcloud_camera_gt_inverse = groundtruth_q_pointcloud_camera * \
                            torch.tensor([-1., -1., -1., 1.], device="cuda")
                        q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
                            (1, 4)), current_q.reshape((1, 4)).cuda(),)
                        q_difference = q_difference.cpu().numpy()
                        angle_difference = np.abs(
                            2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))
                        if angle_difference > np.pi:
                            angle_difference = 2*np.pi - angle_difference
                        errors_q_current_pic.append(angle_difference)

                    # Set the gradient to zero
                    delta_tensor = torch.cat(
                        (delta_tensor_q, delta_tensor_t), axis=0)
                    optimizer_delta_q.zero_grad()
                    optimizer_delta_t.zero_grad()


                    # DEBUG
                    try:
                        # Get current depth map
                        groundtruth_T_pointcloud_camera = torch.tensor(
                            groundtruth_T_pointcloud_camera)
                        visible_points = lidar_measurement.lidar_points_visible(
                            lidar_measurement.point_cloud,
                            groundtruth_T_pointcloud_camera.squeeze(0),
                            resized_camera_info.camera_intrinsics,
                            (resized_camera_info.camera_width, resized_camera_info.camera_height))
                        depth_map = torch.full(
                            (resized_camera_info.camera_height, resized_camera_info.camera_width), -1.0, device="cuda")

                        depth_map = lidar_measurement.lidar_points_to_camera(
                            visible_points,
                            groundtruth_T_pointcloud_camera,
                            resized_camera_info.camera_intrinsics,
                            (resized_camera_info.camera_width,
                                resized_camera_info.camera_height)
                        )
                    except:
                        pass

                    # DEBUG =======================
                    depth_map = resized_depth_image
                    # =============================

                    depth_mask = torch.where(depth_map >= 0, True, False)
                    depth_map = depth_map / torch.max(depth_map)

                    predicted_image, predicted_depth, _, _ = self.rasteriser(
                        GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            delta=delta_tensor,
                            initial_q=initial_q_numpy,
                            initial_t=initial_t_numpy,
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

                    if len(predicted_image.shape) == 3:
                        predicted_image_temp = predicted_image.unsqueeze(0)
                    if len(ground_truth_image.shape) == 3:
                        ground_truth_image_temp = ground_truth_image.unsqueeze(
                            0)

                    L1 = 0.8*torch.abs(predicted_image - ground_truth_image).mean() + 0.2*(1 - ssim(predicted_image_temp, ground_truth_image_temp,
                                                                                                    data_range=1, size_average=True))
                    masked_difference = torch.abs(
                        predicted_depth - depth_map)  # [depth_mask]
                    L_DEPTH = masked_difference.mean()
                    if len(masked_difference) == 0:
                        L_DEPTH = torch.tensor(0)

                    L = L1 + 0.1 * L_DEPTH

                    L.backward()

                    if (not torch.isnan(delta_tensor_q.grad).any()) and (not torch.isnan(delta_tensor_t.grad).any()):
                        optimizer_delta_q.step()
                        optimizer_delta_t.step()

                    if epoch % 5 == 0:
                        scheduler.step()
                        for param_group in optimizer_delta_q.param_groups:
                            if param_group['lr'] < 1e-5:
                                param_group['lr'] = 1e-5
                        scheduler_t.step()
                        for param_group in optimizer_delta_t.param_groups:
                            if param_group['lr'] < 1e-5:
                                param_group['lr'] = 1e-5

                    if epoch % 100 == 0:
                        with torch.no_grad():
                            print(
                                f"============== epoch {epoch} ==========================")
                            print(f"loss:{L}")

                            image_np = predicted_image.cpu().detach().numpy()

                            epsilon = 0.0001
                            delta_numpy_array = delta_tensor.clone().detach().cpu().numpy()
                            current_pose = initial_pose.retract(
                                delta_numpy_array, epsilon=epsilon)
                            current_q, current_t = extract_q_t_from_pose(
                                current_pose)
                            print(f"Initial pose:\n\t{initial_pose}")
                            print(f"Current pose:\n\t{current_pose}")
                            print(f"Ground truth pose:\n\t{pose_groundtruth}")
                            print(f"T_pointcloud_camera_current: ",
                                  T_pointcloud_camera_current)
                            print(f"T_pointcloud_camera_groundtruth: ",
                                  groundtruth_T_pointcloud_camera)

                            im = PIL.Image.fromarray(
                                (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                            if not os.path.exists(os.path.join(self.output_path, f'epochs_delta_{count}/')):
                                os.makedirs(os.path.join(
                                    self.output_path, f'epochs_delta_{count}/'))
                            im.save(os.path.join(self.output_path,
                                    f'epochs_delta_{count}/epoch_{epoch}.png'))
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs_delta_{count}/epoch_{epoch}_q.txt'), current_q.cpu().detach().numpy())
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs_delta_{count}/epoch_{epoch}_t.txt'), current_t.cpu().detach().numpy())

                delta_tensor = torch.cat(
                    (delta_tensor_q, delta_tensor_t), axis=0)
                current_pose = initial_pose.retract(
                            delta_tensor.clone().detach().cpu().numpy(), epsilon=epsilon)
                optimized_poses_lie[count,:] = current_pose.to_tangent().reshape(1,6)
                
                errors_t_current_pic = np.array(errors_t_current_pic)
                errors_t_current_pic = errors_t_current_pic.reshape(
                    (num_epochs + num_coarse_epochs, 3))
                errors_q_current_pic = np.array(errors_q_current_pic)
                errors_q_current_pic = errors_q_current_pic.reshape(
                    (num_epochs + num_coarse_epochs, 1))
                errors_t.append(errors_t_current_pic)
                errors_q.append(errors_q_current_pic)

                plt.plot(errors_q_current_pic[:, 0], color='red', label='x')
                plt.xlabel("Epoch")
                plt.ylabel("Error")
                plt.title("Rotational error")
                plt.legend()
                plt.savefig(os.path.join(
                    self.output_path, f'epochs_delta_{count}/rot_error.png'))
                plt.clf()

                plt.plot(errors_t_current_pic[:, 0], color='red', label='x')
                plt.plot(errors_t_current_pic[:, 1], color='green', label='y')
                plt.plot(errors_t_current_pic[:, 2], color='blue', label='z')
                plt.xlabel("Epoch")
                plt.ylabel("Error")
                plt.title("Translational error")
                plt.legend()
                plt.savefig(os.path.join(
                    self.output_path, f'epochs_delta_{count}/trasl_error.png'))
                plt.clf()

                np.savetxt(os.path.join(
                    self.output_path, f'epochs_delta_{count}/error_q.out'), errors_q_current_pic, delimiter=',')
                np.savetxt(os.path.join(
                    self.output_path, f'epochs_delta_{count}/error_t.out'), errors_t_current_pic, delimiter=',')

            errors_q_numpy = np.array(errors_q)
            errors_t_numpy = np.array(errors_t)
            errors_q_mean = np. mean(
                errors_q_numpy, axis=0)  # epochs x 1 array
            errors_t_mean = np. mean(
                errors_t_numpy, axis=0)  # epochs x 3 array

            plt.plot(errors_q_mean[:, 0])
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.title("Rotational error")
            plt.savefig(
                os.path.join(self.output_path, "rot_error.png"))
            plt.clf()

            plt.plot(errors_t_mean[:, 0])
            plt.plot(errors_t_mean[:, 1])
            plt.plot(errors_t_mean[:, 2])
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.title("Translational error")
            plt.savefig(
                os.path.join(self.output_path, "trasl_error.png"))
            plt.clf()

            # Save groundtruth and estimate
            np.save(os.path.join(self.output_path, "optimized_poses_lie.out"), optimized_poses_lie)
            np.save(os.path.join(self.output_path, "groundtruth_poses_lie.out"), groundtruth_poses_lie)

def main():
    parser = argparse.ArgumentParser(description='Parquet file path')
    parser.add_argument('--parquet_path', type=str, help='Parquet file path')
    # parser.add_argument('--images-path', type=str, help='Images file path')
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

if __name__ == '__main__':
    main()