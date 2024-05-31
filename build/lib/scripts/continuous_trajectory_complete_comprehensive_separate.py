import continuous_trajectory

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
import pypose as pp
from pytorch_msssim import ssim
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
import curve_evaluation

from scipy.interpolate import splev, splrep

_EPS = 1e-8


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
        # self.output_path = "scripts/continuous_trajectory_output_q_t_perturbed_z_spline_sull_tra"
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16
        self.groundtruth_trajectory_path = "scripts/data/val_path.obj"

        # 3D plot
        self.f1 = pl.figure(1)
        self.a3d = self.f1.add_subplot(111, projection='3d')

        # 2D plot
        self.f2 = pl.figure(2)
        self.a2d = self.f2.add_subplot(111)

        # Load blender trajectory
        with open(self.groundtruth_trajectory_path, 'r') as obj_file:
            lines = obj_file.readlines()
        vertices = []
        for line in lines:
            if line.startswith('v '):
                vertex = line.split()[1:]
                vertex = [float(coord) for coord in vertex]
                vertices.append(vertex)
        self.vertices_array = np.array(vertices)

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

    def extract_data(self, d):
        ground_truth_image_torch_list = []
        groundtruth_T_pointcloud_camera_torch_list = []
        perturbed_T_pointcloud_camera_torch_list = []
        ground_truth_image_downsampled_list = []
        resized_depth_image_list = []
        resized_depth_image_downsampled_list = []

        # Initialize zspline with noisy poses estimate
        for view_dict in d:
            ground_truth_image_path = view_dict["image_path"]
            # Load the image and convert it to a numpy array
            ground_truth_image_numpy = np.array(
                PIL.Image.open(ground_truth_image_path))
            ground_truth_image_tensor = torchvision.transforms.functional.to_tensor(
                ground_truth_image_numpy)

            depth_image_path = view_dict["depth_path"]
            # Load the image and convert it to a numpy array
            depth_image_numpy = np.array(
                PIL.Image.open(depth_image_path))
            depth_image_tensor = torchvision.transforms.functional.to_tensor(
                depth_image_numpy)
            # Append the image tensor to a list

            ground_truth_images, resized_camera_info, resized_depth_image = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image_tensor,
                                                                                                                                        depth_image_tensor,
                                                                                                                                        self.camera_info,
                                                                                                                                        1)
            ground_truth_image_downsampled, resized_camera_info_downsampled, resized_depth_image_downsampled = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image_tensor,
                                                                                                                                                                           depth_image_tensor,
                                                                                                                                                                           self.camera_info,
                                                                                                                                                                           2)
            ground_truth_images = ground_truth_images.cuda()
            ground_truth_image_torch_list.append(ground_truth_images)
            resized_depth_image = resized_depth_image.cuda()
            resized_depth_image_list.append(resized_depth_image)
            ground_truth_image_downsampled = ground_truth_image_downsampled.cuda()
            ground_truth_image_downsampled_list.append(
                ground_truth_image_downsampled)
            resized_depth_image_downsampled = resized_depth_image_downsampled.cuda()
            resized_depth_image_downsampled_list.append(
                resized_depth_image_downsampled)

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
        return ground_truth_image_torch_list, groundtruth_T_pointcloud_camera_torch_list, perturbed_T_pointcloud_camera_torch_list, resized_camera_info, resized_depth_image_list, \
            ground_truth_image_downsampled_list, resized_camera_info_downsampled, resized_depth_image_downsampled_list

    # numpy_spline_knots: FIRST t, THEN q
    def plot_trajectory(self, numpy_spline_knots, color="black"):

        # self.a3d.plot(self.vertices_array[:, 0], -self.vertices_array[:, 2],
        #               self.vertices_array[:, 1], color="green", label="Blender path")
        # self.a2d.plot(self.vertices_array[:, 0], -self.vertices_array[:, 2],
        #               color="green", label="Blender path")

        for i in range(numpy_spline_knots.shape[0]-3):
            plotCoordinateFrame.plot_trajectory(
                self.a3d, numpy_spline_knots[i:i+4, :3], color=color, linewidth=1, label="estimation", evaluate_zspline=True)
            plotCoordinateFrame.plot_trajectory_2d(
                self.a2d, numpy_spline_knots[i:i+4, :3], color=color, linewidth=1, label="estimation", evaluate_zspline=True)

        self.a2d.grid()
        self.a3d.grid()
        self.a2d.axis("equal")
        self.a3d.axis("equal")

    def start(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.rasteriser = GaussianPointCloudContinuousPoseRasterisation(
            config=GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationConfig(
                near_plane=0.001,
                far_plane=1000.,
                depth_to_sort_key_scale=100.,
                enable_depth_grad=True,
            ))

        with open(self.config.json_file_path) as f:
            d = json.load(f)
            N = len(d)
            batch_size = 1 #2
            num_epochs = 10001

            num_segments = N//batch_size

            # Reshape data to fit batch_size
            N = num_segments*batch_size
            d = d[0:N]
            print(f"Reshaped N:{N}")

            # Save index in batch for each frame
            index_in_segment = np.array([i % batch_size for i in range(N)])
            time_index = index_in_segment*(1/batch_size)
            ground_truth_image_torch_list, groundtruth_T_pointcloud_camera_torch_list, perturbed_T_pointcloud_camera_torch_list, resized_camera_info, resized_depth_image_list, \
                ground_truth_image_downsampled, resized_camera_info_downsampled, resized_depth_image_downsampled = self.extract_data(
                    d)

            groundtruth_T_pointcloud_camera_torch = torch.stack(
                groundtruth_T_pointcloud_camera_torch_list, dim=0).squeeze(
                1)
            groundtruth_q_pointcloud_camera, groundtruth_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                groundtruth_T_pointcloud_camera_torch)

            perturbed_T_pointcloud_camera_torch = torch.stack(
                perturbed_T_pointcloud_camera_torch_list, dim=0).squeeze(1)
            perturbed_q_pointcloud_camera, perturbed_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                perturbed_T_pointcloud_camera_torch)

            groundtruth_lie = np.zeros((N, 6))
            perturbed_lie = np.zeros((N, 6))
            groundtruth_t_pointcloud_camera = groundtruth_t_pointcloud_camera.cpu().numpy()

            for j in range(N):
                rotation_groundtruth = sym.Rot3(
                    groundtruth_q_pointcloud_camera[j].cpu().numpy())
                pose_groundtruth = sym.Pose3(
                    R=rotation_groundtruth, t=groundtruth_t_pointcloud_camera[j])
                groundtruth_lie[j, :] = (pose_groundtruth.to_tangent(_EPS))

                perturbed_rotation = sym.Rot3(
                    perturbed_q_pointcloud_camera[j].cpu().numpy())
                pose_perturbed = sym.Pose3(
                    R=perturbed_rotation, t=perturbed_t_pointcloud_camera[j].cpu().numpy())
                perturbed_lie[j, :] = (pose_perturbed.to_tangent(_EPS))

            bases = curve_evaluation.evaluate_spline_bases_lsq(
                perturbed_lie, batch_size, enable_zspline=True)

            groundtruth_bases = curve_evaluation.evaluate_spline_bases_lsq(
                groundtruth_lie, batch_size, enable_zspline=True
            )
            
            groundtruth_bases = groundtruth_lie
            bases = groundtruth_bases + \
                np.random.normal(loc=0, scale=0.02, size=groundtruth_lie.shape)
                
            # I have a problem with last values of bases; cut the trajectory
            num_segments = num_segments-1 #num_segments-1
            N = num_segments*batch_size
            d = d[0:N]
            time_index = time_index[0:N]
            index_in_segment = index_in_segment[0:N]
            index_of_segment = np.array(
                range(num_segments)).repeat(batch_size, 0)
            print(f"Reshaped N:{N}")

            # bases = bases[:(4+(num_segments-1)), :]
            bases = bases[:num_segments, :]
            num_segments = num_segments-3
            errors_t = np.zeros((N, num_epochs))
            errors_q = torch.zeros((N, num_epochs))

            # Plot initial trajectory guess
            self.plot_trajectory(
                np.hstack((bases[:, 3:], bases[:, :3])), color="orange")

            self.a2d.scatter(
                bases[:, 3], bases[:, 4], color="orange", s=2)
            self.a3d.scatter(
                bases[:, 3], bases[:, 4], bases[:, 5], color="orange", s=2)

            self.a2d.scatter(
                perturbed_t_pointcloud_camera[:, 0].cpu().numpy(), perturbed_t_pointcloud_camera[:, 1].cpu().numpy(), color="yellow", s=2)
            self.a3d.scatter(
                perturbed_t_pointcloud_camera[:, 0].cpu().numpy(), perturbed_t_pointcloud_camera[:, 1].cpu().numpy(), perturbed_t_pointcloud_camera[:, 2].cpu().numpy(), color="yellow", s=2)

            self.a2d.scatter(
                groundtruth_t_pointcloud_camera[:, 0], groundtruth_t_pointcloud_camera[:, 1], color="red", s=2)
            self.a3d.scatter(
                groundtruth_t_pointcloud_camera[:, 0], groundtruth_t_pointcloud_camera[:, 1], groundtruth_t_pointcloud_camera[:, 2], color="red", s=2)

            # for i in range(N):
            #     self.a2d.text(
            #     groundtruth_t_pointcloud_camera[i, 0].cpu().numpy(), groundtruth_t_pointcloud_camera[i, 1].cpu().numpy(),f"{i}", color="red",  fontsize=12)

            self.a2d.grid()
            self.a3d.grid()
            self.f1.savefig(os.path.join(
                self.output_path, f'Initial_guess_3d.png'))
            self.f2.savefig(os.path.join(
                self.output_path, f'Initial_guess_2d.png'))

            # Start processing of data, frame after frame
            # Setup optimization
            pypose_bspline_knots = torch.zeros((bases.shape[0], 7))
            for base_number, base in enumerate(bases):
                base_lie = sym.Pose3.from_tangent(base)
                pypose_bspline_knots[base_number, :] = torch.hstack((torch.tensor([
                    base_lie.position()]).to(torch.float32), torch.tensor([
                        base_lie.rotation().data[:]]).to(torch.float32)))

            pypose_bspline_knots_t = pypose_bspline_knots[:, :3]
            pypose_bspline_knots_q = pypose_bspline_knots[:, 3:]

            pypose_bspline_knots_t.requires_grad_()
            pypose_bspline_knots_q.requires_grad_()

            pypose_segment_base_t = [torch.zeros(
                (4, 3)) for s in range(num_segments)]
            pypose_segment_base_q = [torch.zeros(
                (4, 3)) for s in range(num_segments)]
            optimizer_bspline_bases_t = []
            optimizer_bspline_bases_q = []
            scheduler_t = []
            scheduler_q = []

            print(num_segments)
            for s in range(num_segments):
                # Copy t and q as starting point for optimization
                # How to ensure q and t are the same for consecutive optimizations of segments?
                pypose_segment_base_t[s] = pypose_bspline_knots_t[s:s +
                                                                  4, :].clone().detach()
                pypose_segment_base_q[s] = pypose_bspline_knots_q[s:s +
                                                                  4, :].clone().detach()

                pypose_segment_base_t[s].requires_grad_()
                pypose_segment_base_q[s].requires_grad_()
                optimizer_bspline_bases_t.append(torch.optim.Adam(
                    [pypose_segment_base_t[s]], lr=5e-5, betas=(0.9, 0.999)))  # 1e-4, 1e-4 for z-spline without depth
                optimizer_bspline_bases_q.append(torch.optim.Adam(
                    [pypose_segment_base_q[s]], lr=5e-5, betas=(0.9, 0.999)))

                scheduler_t.append(torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_bspline_bases_t[s], gamma=0.9947))  # optimizer_bspline_bases_t[s]
                scheduler_q.append(torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer_bspline_bases_q[s], gamma=0.9947))

            # for epoch in range(num_epochs):

                # with torch.no_grad():
                # for n in range(N):
                #     errors_t[n, epoch] = np.linalg.norm(
                #         np.array(current_pose[n, :3]) - np.array(groundtruth_t_pointcloud_camera[n, :]).reshape((1, 3)))
                #     errors_q[n, epoch] = continuous_trajectory.quaternion_difference_rad(torch.tensor(
                #         current_pose[n, 3:]), torch.tensor(groundtruth_q_pointcloud_camera[n, :]).reshape((1, 4)))

            total_count = 0

            for epoch in range(num_epochs):
                for segment in range(num_segments):
                    print("Current segment:", segment)
                    # print(pypose_bspline_knots_t[segment:segment+4, :].data)
                    # pypose_segment_base_t = pypose_bspline_knots_t[segment:segment+4, :]
                    # pypose_segment_base_q = pypose_bspline_knots_q[segment:segment+4, :]
                    # pypose_segment_base_t.requires_grad_()
                    # pypose_segment_base_q.requires_grad_()
                    # optimizer_bspline_bases_t_segment = torch.optim.Adam(
                    #     [pypose_segment_base_t], lr=1e-3, betas=(0.9, 0.999))  # 1e-4, 1e-4 for z-spline without depth
                    # optimizer_bspline_bases_q_segment = torch.optim.Adam(
                    #     [pypose_segment_base_q], lr=1e-3, betas=(0.9, 0.999))
                    # scheduler_t_segment = (torch.optim.lr_scheduler.ExponentialLR(
                    #     optimizer=optimizer_bspline_bases_t_segment, gamma=0.9947))
                    # scheduler_q_segment = torch.optim.lr_scheduler.ExponentialLR(
                    #     optimizer=optimizer_bspline_bases_q_segment, gamma=0.9947)

                    L1 = 0
                    total_count += 1
                    for ss in range(num_segments):
                        pypose_segment_base_t[ss].data.copy_(pypose_bspline_knots_t[ss:ss +
                                            4].data)
                        pypose_segment_base_q[ss].data.copy_(pypose_bspline_knots_q[ss:ss +
                                            4].data)
                        
                    for i in range(batch_size):

                        current_index = segment*batch_size+i

                        # segment = index_of_segment[current_index]

                        # for i in range(batch_size):
                        optimizer_bspline_bases_t[segment].zero_grad()
                        optimizer_bspline_bases_q[segment].zero_grad()

                        timestamp = time_index[current_index]

                        print("pypose_segment_base_t:", pypose_segment_base_t)
                        current_pose_tensor = curve_evaluation.cubic_bspline_interpolation(
                            pp.SE3(torch.hstack(
                                (pypose_segment_base_t[segment], pypose_segment_base_q[segment])).double()),
                            u=torch.tensor([timestamp]).double(),
                            enable_z_spline=True)
                        if segment == 0:
                            print(current_index)
                            print(timestamp)
                            print(current_pose_tensor)
                            print(pypose_segment_base_t)
                        current_pose_tensor = current_pose_tensor.cuda()
                        current_pose_tensor.requires_grad_()
                        current_pose_tensor.retain_grad()
                        current_pose = current_pose_tensor.clone().detach().cpu().numpy()
                        with torch.no_grad():
                            errors_t[current_index, epoch] = np.linalg.norm(
                                np.array(current_pose[:, :3]) - np.array(groundtruth_t_pointcloud_camera[current_index + 1, :]).reshape((1, 3)))
                            errors_q[current_index, epoch] = continuous_trajectory.quaternion_difference_rad(torch.tensor(
                                current_pose[:, 3:]), torch.tensor(groundtruth_q_pointcloud_camera[current_index + 1, :]).reshape((1, 4)))

                        if epoch < 0:  # coarse alignment
                            camera_info = resized_camera_info_downsampled
                            ground_truth_image = ground_truth_image_downsampled[current_index+1]
                            ground_truth_image_temp = ground_truth_image
                            depth_map = resized_depth_image_downsampled[i]
                        else:
                            camera_info = resized_camera_info
                            ground_truth_image = ground_truth_image_torch_list[current_index+1]
                            depth_map = resized_depth_image_list[i]

                        predicted_image, predicted_depth, _, _ = self.rasteriser(
                            GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationInput(
                                point_cloud=self.scene.point_cloud,
                                point_cloud_features=self.scene.point_cloud_features,
                                point_invalid_mask=self.scene.point_invalid_mask,
                                point_object_id=self.scene.point_object_id,
                                camera_info=camera_info,
                                current_pose=current_pose_tensor.float(),
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

                        L1 += 0.8*torch.abs(predicted_image_temp - ground_truth_image_temp).mean() + 0.2*(1 - ssim(predicted_image_temp, ground_truth_image_temp,
                                                                                                                   data_range=1, size_average=True))
                        depth_map = depth_map / torch.max(depth_map)
                        depth_map = depth_map.squeeze(0)
                        masked_difference = torch.abs(
                            predicted_depth - depth_map)  # [depth_mask]
                        # L_DEPTH = masked_difference.mean()
                        # if len(masked_difference) == 0:
                        #     L_DEPTH = torch.tensor(0)

                        # L1 += L_DEPTH
                    L1.backward(retain_graph=True)

                    dL1_d_pypose_knots_t = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                        pypose_segment_base_t[segment]], grad_outputs=current_pose_tensor.grad, retain_graph=True)

                    dL1_d_pypose_knots_q = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                        pypose_segment_base_q[segment]], grad_outputs=current_pose_tensor.grad, retain_graph=True)

                    # print ("Gradient:", dL1_d_pypose_knots_t)
                    # print("before")
                    # print(f"segment:{segment}")
                    # # print(current_pose_tensor)
                    # # print(dL1_d_pypose_knots_t)
                    # print(pypose_segment_base_t)
                    # print(pypose_segment_base_t[segment].grad)
                    
                    if (dL1_d_pypose_knots_t) is not None:
                        pypose_segment_base_t[segment].grad = dL1_d_pypose_knots_t[0]
                        if (not torch.isnan(pypose_segment_base_t[segment].grad).any()):
                            if (not torch.isinf(pypose_segment_base_t[segment].grad).any()):
                                print("Stepping")
                                optimizer_bspline_bases_t[segment].step()
                                optimizer_bspline_bases_t[segment].zero_grad(
                                )
                    # print("after")
                    # print(pypose_segment_base_t)

                    if (dL1_d_pypose_knots_q) is not None:
                        pypose_segment_base_q[segment].grad = dL1_d_pypose_knots_q[0]
                        if (not torch.isnan(pypose_segment_base_q[segment].grad).any()):
                            if (not torch.isinf(pypose_segment_base_q[segment].grad).any()):
                                optimizer_bspline_bases_q[segment].step()
                                optimizer_bspline_bases_q[segment].zero_grad(
                                )

                    # Update global segments
                    pypose_bspline_knots_t[segment:segment +
                                        4].data.copy_(pypose_segment_base_t[segment].data)
                    pypose_bspline_knots_q[segment:segment +
                                        4].data.copy_(pypose_segment_base_q[segment].data)
                    for ss in range(num_segments):
                        pypose_segment_base_t[ss].data.copy_(pypose_bspline_knots_t[ss:ss +
                                            4].data)
                        pypose_segment_base_q[ss].data.copy_(pypose_bspline_knots_q[ss:ss +
                                            4].data)
                if epoch % 5 == 0:
                    # for s in range(num_segments):
                    scheduler_t[segment].step()
                    for param_group in optimizer_bspline_bases_t[segment].param_groups:
                        if param_group['lr'] < 1e-6:
                            param_group['lr'] = 1e-6

                if epoch % 5 == 0:
                    # for s in range(num_segments):
                    scheduler_q[segment].step()
                    for param_group in optimizer_bspline_bases_q[segment].param_groups:
                        if param_group['lr'] < 1e-5:
                            param_group['lr'] = 1e-5

                    # try:
                    #     if segment > 0:
                    #         pypose_segment_base_t[segment-1].data[1:-1, :].copy_(
                    #             pypose_segment_base_t[segment].data[0:2, :])
                    #     if segment < num_segments-1:
                    #         pypose_segment_base_t[segment+1].data[0:2, :].copy_(
                    #             pypose_segment_base_t[segment].data[1:-1, :])
                    #     print(pypose_segment_base_t[segment+1])
                    # except Exception as e:
                    #     print("Exception")
                    #     print(e)
                    # pypose_bspline_knots_t[segment:segment +
                    #                        4].data.copy_(pypose_segment_base_t[segment].data)
                    # pypose_bspline_knots_q[segment:segment +
                    #                        4].data.copy_(pypose_segment_base_q[segment].data)
                    # Plot trajectory result
                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} - current photometric loss: {L1}")
                    numpy_spline_knots = np.hstack(
                        (pypose_bspline_knots_t.clone().detach().cpu().numpy(), pypose_bspline_knots_q.clone().detach().cpu().numpy()))
                    print(f"Numpy zspline knots: {numpy_spline_knots}")
                    # Plot full trajectory ===============================
                    self.a2d.clear()
                    self.a3d.clear()
                    self.a2d = self.f2.add_subplot(111)

                    self.plot_trajectory(
                        np.hstack((bases[:, 3:], bases[:, :3])), color="orange")

                    self.a3d.plot(self.vertices_array[:, 0], -self.vertices_array[:, 2],
                                    self.vertices_array[:, 1], color="green", label="Blender path", linewidth=0.5)
                    self.a2d.plot(self.vertices_array[:, 0], -self.vertices_array[:, 2],
                                    color="green", label="Blender path", linewidth=0.5)
                    self.a2d.scatter(
                        bases[:, 3], bases[:, 4], color="orange", s=1, alpha=0.7)
                    self.a3d.scatter(
                        bases[:, 3], bases[:, 4], bases[:, 5], color="orange", s=1, alpha=0.7)

                    # self.plot_trajectory(numpy_spline_knots)
                    for s in range(num_segments):
                        current_bases = np.hstack(
                            (pypose_segment_base_t[s].clone().detach().cpu().numpy(), pypose_segment_base_q[s].clone().detach().cpu().numpy()))
                        self.plot_trajectory(current_bases)
                    self.a2d.scatter(
                        numpy_spline_knots[:, 0], numpy_spline_knots[:, 1], color="gray", s=1.5)
                    self.a3d.scatter(
                        numpy_spline_knots[:, 0], numpy_spline_knots[:, 1], numpy_spline_knots[:, 2], color="gray", s=1.5)
                    self.a2d.scatter(
                        groundtruth_t_pointcloud_camera[:, 0], groundtruth_t_pointcloud_camera[:, 1], color="red", s=2)
                    self.a3d.scatter(
                        groundtruth_t_pointcloud_camera[:, 0], groundtruth_t_pointcloud_camera[:, 1], groundtruth_t_pointcloud_camera[:, 2], color="red", s=2)

                    self.a2d.axis("equal")
                    self.a3d.axis("equal")

                    self.a2d.grid(visible=True)
                    self.a3d.grid(visible=True)
                    self.f2.savefig(os.path.join(
                        self.output_path, f'full_path_2d__epoch_{total_count}.png'), dpi=600)

                    self.a2d.clear()
                    self.a3d.clear()

                    for n in range(N):
                        # for k in range(batch_size):
                        # k_current = segment*batch_size+k
                        plt.clf()
                        plt.plot(errors_q[n, :epoch])
                        plt.xlabel("Epoch")
                        plt.ylabel("Error")
                        plt.title("Rotational error")

                        plt.savefig(
                            os.path.join(self.output_path, f"final_rot_error_frame_{n}.png"))
                        plt.clf()

                        plt.plot(errors_t[n, :epoch])
                        plt.xlabel("Epoch")
                        plt.ylabel("Error")
                        plt.title("Translational error")

                        plt.savefig(
                            os.path.join(self.output_path, f"final_trans_error_frame_{n}.png"))
                        plt.clf()

                        # Plot mean
                        plt.clf()
                        plt.plot(torch.mean(errors_q[:, :epoch], axis=0))
                        plt.xlabel("Epoch")
                        plt.ylabel("Error")
                        plt.title("Mean Rotational error")

                        plt.savefig(
                            os.path.join(self.output_path, f"final_rot_error_mean.png"))
                        plt.clf()

                        plt.plot(np.mean(errors_t[:, :epoch], axis=0))
                        plt.xlabel("Epoch")
                        plt.ylabel("Error")
                        plt.title("Mean Translational error")

                        plt.savefig(
                            os.path.join(self.output_path, f"final_trans_error_mean.png"))
                        plt.clf()
                # pypose_bspline_knots_t[segment:segment +
                #                        4].data.copy_(pypose_segment_base_t[segment].data)
                # pypose_bspline_knots_q[segment:segment +
                #                        4].data.copy_(pypose_segment_base_q[segment].data)


def main():
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


if __name__ == '__main__':
    main()
