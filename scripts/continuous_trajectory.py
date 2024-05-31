
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


def interpolate_bspline(time: float, bases) -> torch.Tensor:
    tt = torch.pow(time, torch.arange(0, 4, device="cuda"))
    w = torch.matmul(M, tt)
    if isinstance(bases, (np.ndarray)):
        w = w.detach().clone().cpu().numpy()
    delta_pose = bases[0] + w[0]*(bases[1, :] - bases[0, :]) + w[1]*(
        bases[2, :] - bases[1, :]) + w[2]*(bases[3, :]-bases[2, :])

    return delta_pose


def evaluate_spline_bases(poses: np.array):  # poses: Nx6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(), tt)

        section = np.array([1-w[0], w[0]-w[1], w[1]-w[2], w[2]]
                           ).reshape((1, 4))  # .repeat(6, axis=0)
        A[i, :] = section

    bases, _, _, _ = np.linalg.lstsq(A, poses)
    return np.array(bases).reshape(4, 6)


def pose_error(reference: np.array, test: np.array):
    pose_reference = sym.Pose3.from_tangent(reference)
    pose_test = sym.Pose3.from_tangent(test)
    error_t = np.linalg.norm(
        np.array(pose_test.position()) - np.array(pose_reference.position()))
    error_q = quaternion_difference_rad(torch.tensor(
        [pose_test.rotation().data[:]]).to(torch.float32), torch.tensor(
        [pose_reference.rotation().data[:]]).to(torch.float32))
    return error_q, error_t


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
        self.output_path = "scripts/continuous_trajectory_output_q_t_perturbed_z_spline_lidar"
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
        lambdas_depth= np.array([0.]) #0.001, 0.01, 0.05, 0.1, 0.5, 1.0
        

        with open(self.config.json_file_path) as f:
            d = json.load(f)

            total_images = len(d)
            batch_size = 2  # Depends on how many measurements in time - straightness of trajectory

            num_batches = total_images//batch_size
            bases = torch.zeros((4+(num_batches-1), 7))

            if total_images % batch_size != 0:
                print("Discarding images")

            # Batch images
            for i in range(total_images):
                if i == 0 or i>1:
                    continue
                for lambda_depth in lambdas_depth:
                    self.output_path = f"scripts/continuous_trajectory_output_q_t_perturbed_test"
                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                    num_segment = i//batch_size
                    index_in_segment = i - num_segment*batch_size
                    print(
                        f"Frame {i} - index {index_in_segment} in segment {num_segment}")
                    start_t_optimization = True
                    print("Setting start_t_optimization to True")
                    if not os.path.exists(os.path.join(self.output_path, f"pickle_files_batch_{i}")):
                        os.makedirs(os.path.join(self.output_path,
                                    f"pickle_files_batch_{i}"))
                    if not os.path.exists(os.path.join(self.output_path, f"2d_plots_batch_{i}")):
                        os.makedirs(os.path.join(
                            self.output_path, f"2d_plots_batch_{i}"))
                    if not os.path.exists(os.path.join(self.output_path, f"3d_plots_batch_{i}")):
                        os.makedirs(os.path.join(
                            self.output_path, f"3d_plots_batch_{i}"))
                    print(
                        f"=================Batch {i}========================")

                    view = d[i:i+batch_size]

                    ground_truth_image_torch_list = []
                    groundtruth_T_pointcloud_camera_torch_list = []
                    perturbed_T_pointcloud_camera_torch_list = []
                    depth_image_torch_list = []
                    
                    for view_dict in view:
                        # Load groundtruth image path from the current dictionary
                        ground_truth_image_path = view_dict["image_path"]
                        print(f"Loading image {ground_truth_image_path}")

                        # Load the image and convert it to a numpy array
                        ground_truth_image_numpy = np.array(
                            PIL.Image.open(ground_truth_image_path))
                        ground_truth_image_tensor = torchvision.transforms.functional.to_tensor(
                            ground_truth_image_numpy)

                        # DEBUG =============================================
                        depth_image_path = view_dict["depth_path"]
                        print(f"Loading image {depth_image_path}")
                        depth_image_path_numpy = np.array(
                            PIL.Image.open(depth_image_path))
                        depth_image = torchvision.transforms.functional.to_tensor(
                            depth_image_path_numpy)
                        # ===============================================

                        ground_truth_images, resized_camera_info, resized_depth_image = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image_tensor,
                                                                                                                                                    depth_image,
                                                                                                                                                    self.camera_info,
                                                                                                                                                    1)
                        ground_truth_images = ground_truth_images.cuda()
                        ground_truth_image_torch_list.append(ground_truth_images)

                        resized_depth_image = resized_depth_image.cuda()
                        depth_image_torch_list.append(resized_depth_image)
                        
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
                    # DEBUG start from noisy base guesses
                    # perturbed_bases = evaluate_spline_bases(np.array(
                    #     [perturbed_pose[k].to_tangent(_EPS)
                    #      for k in range(batch_size)]
                    # ))

                    # groundtruth_bases = evaluate_spline_bases(np.array(  # FIX
                    #     [groundtruth_pose[k].to_tangent(
                    #         _EPS) for k in range(batch_size)]
                    # ))
                    
                    groundtruth_bases = curve_evaluation.evaluate_spline_bases_lsq(np.array(  
                        [perturbed_pose[k].to_tangent( #groundtruth_pose
                            _EPS) for k in range(batch_size)]
                    ), batch_size, enable_zspline=True)
                    # DEBUG
                    # groundtruth_bases = curve_evaluation.evaluate_z_spline_bases(np.array(
                    #     [groundtruth_pose[k].to_tangent(
                    #         _EPS) for k in range(batch_size)]
                    # ))

                    if i == 0:
                        perturbed_bases = groundtruth_bases + \
                            np.random.normal(loc=0, scale=0.05, size=(
                                4, 6))  # np.hstack((np.zeros((4,3)), np.random.normal(loc=0, scale=0.05, size=(4, 3))))
                        bspline_bases = torch.tensor(perturbed_bases).reshape(
                            (4, 6)).cuda()

                        # DEBUG - set bspine bases as pypose lietensor (pose, angle)
                        pypose_bspline_knots = torch.zeros((4, 7))
                        for base_number, base in enumerate(bspline_bases):
                            base_lie = sym.Pose3.from_tangent(base)
                            pypose_bspline_knots[base_number, :] = torch.hstack((torch.tensor([
                                base_lie.position()]).to(torch.float32), torch.tensor([
                                    base_lie.rotation().data[:]]).to(torch.float32)))
                    else:
                        # pypose_bspline_knots = torch.zeros((4, 7))
                        # pypose_bspline_knots[:3, :] = bases[i:i+3, :]

                        # # Translation guess:set to current velocity
                        # pypose_bspline_knots[3, :3] = bases[i+2,
                        #                                     :3]  # + (bases[i+2, :3]-bases[i+1, :3])
                        # # Rotation: set as last estimate
                        # pypose_bspline_knots[3, 3:] = bases[i+2, 3:]
                        perturbed_bases = groundtruth_bases + \
                            np.random.normal(loc=0, scale=0.05, size=(
                                4, 6))  # np.hstack((np.zeros((4,3)), np.random.normal(loc=0, scale=0.05, size=(4, 3))))
                        bspline_bases = torch.tensor(perturbed_bases).reshape(
                            (4, 6)).cuda()
                        pypose_bspline_knots = torch.zeros((4, 7))
                        for base_number, base in enumerate(bspline_bases):
                            base_lie = sym.Pose3.from_tangent(base)
                            pypose_bspline_knots[base_number, :] = torch.hstack((torch.tensor([
                                base_lie.position()]).to(torch.float32), torch.tensor([
                                    base_lie.rotation().data[:]]).to(torch.float32)))

                    pypose_groundtruth_bspline_knots = torch.zeros((4, 7))
                    for base_number, base in enumerate(groundtruth_bases):
                        base_lie = sym.Pose3.from_tangent(base)
                        pypose_groundtruth_bspline_knots[base_number, :] = torch.hstack((torch.tensor([
                            base_lie.position()]).to(torch.float32), torch.tensor([
                                base_lie.rotation().data[:]]).to(torch.float32)))

                    # pypose_bspline_knots.requires_grad_()
                    pypose_bspline_knots_t = pypose_bspline_knots[:, :3]
                    pypose_bspline_knots_q = pypose_bspline_knots[:, 3:]

                    pypose_bspline_knots_t.requires_grad_()
                    pypose_bspline_knots_q.requires_grad_()
                    self.rasteriser = GaussianPointCloudContinuousPoseRasterisation(
                        config=GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationConfig(
                            near_plane=0.001,
                            far_plane=1000.,
                            depth_to_sort_key_scale=100.,
                            enable_depth_grad=True,
                        ))

                    optimizer_bspline_bases_t = torch.optim.Adam(
                        [pypose_bspline_knots_t], lr=1e-3, betas=(0.9, 0.999)) #1e-4, 1e-4 for z-spline without depth
                    optimizer_bspline_bases_q = torch.optim.Adam(
                        [pypose_bspline_knots_q], lr=1e-3, betas=(0.9, 0.999))  

                    scheduler_q = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer_bspline_bases_q, gamma=0.9947)#
                    scheduler_t = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer_bspline_bases_t, gamma=0.9947)

                    num_epochs = 3001

                    errors_t = np.zeros((batch_size, num_epochs))
                    errors_q = np.zeros((batch_size, num_epochs))
                    error_bases_q = np.zeros((4, num_epochs))
                    error_bases_t = np.zeros((4, num_epochs))

                    for epoch in range(num_epochs):
                        L1 = 0
                        for v, view_dict in enumerate(view):
                            # Set the gradient to zero
                            # optimizer_bspline_bases.zero_grad()
                            optimizer_bspline_bases_q.zero_grad()
                            optimizer_bspline_bases_t.zero_grad()

                            current_t = v / batch_size  # Since I assume measurements equally spaced in time

                            pypose_bspline_knots = torch.hstack(
                                (pypose_bspline_knots_t, pypose_bspline_knots_q))
                            pypose_bspline_knots.requires_grad_()
                            pypose_bspline_knots.retain_grad()

                            current_pose_tensor = curve_evaluation.cubic_bspline_interpolation(
                                pp.SE3(torch.hstack((pypose_bspline_knots_t, pypose_bspline_knots_q))), torch.tensor([current_t]), enable_eps=False,enable_z_spline=True )  # 

                            # current_pose_tensor = curve_evaluation.bspline(
                            #     pp.SE3(torch.hstack((pypose_bspline_knots_t, pypose_bspline_knots_q)).unsqueeze(0)), torch.tensor([current_t],device="cpu"),enable_z_spline=True).squeeze(0)

                            current_pose_tensor = current_pose_tensor.cuda()
                            current_pose_tensor.requires_grad_()
                            current_pose_tensor.retain_grad()
                            current_pose = current_pose_tensor.clone().detach().cpu().numpy()

                            # Plot pose error
                            with torch.no_grad():
                                gt_pose = curve_evaluation.cubic_bspline_interpolation(
                                    pp.SE3(pypose_groundtruth_bspline_knots), torch.tensor([current_t]), enable_eps=False, enable_z_spline=True)  # 

                                # gt_pose = curve_evaluation.bspline(
                                #     pp.SE3(pypose_groundtruth_bspline_knots.unsqueeze(0)), torch.tensor([current_t],device="cpu"),enable_z_spline=True).squeeze(0)

                                errors_t[v, epoch] = np.linalg.norm(
                                    np.array(current_pose[0, :3]) - np.array(gt_pose[0, :3]))
                                errors_q[v, epoch] = quaternion_difference_rad(torch.tensor(
                                    current_pose[0, 3:]), torch.tensor(gt_pose[0, 3:]))
                                
                                # print(f"Error t: \n\t {errors_t[i,epoch]}\nError q: \n\t{errors_q[i,epoch]}")

                                if np.isnan(errors_t[v, epoch]):
                                    print(current_pose[0, :3])
                                    print(pypose_groundtruth_bspline_knots)

                                for j in range(4):
                                    # DEBUG
                                    error_t = np.linalg.norm(
                                        np.array(pypose_bspline_knots[j, :3]) - np.array(pypose_groundtruth_bspline_knots[j, :3]))
                                    error_q = quaternion_difference_rad(torch.tensor(pypose_bspline_knots[j, 3:]), torch.tensor(
                                        pypose_groundtruth_bspline_knots[j, 3:]))
                                    error_bases_q[j, epoch] = error_q
                                    error_bases_t[j, epoch] = error_t

                            predicted_image, predicted_depth, _, _ = self.rasteriser(
                                GaussianPointCloudContinuousPoseRasterisation.GaussianPointCloudContinuousPoseRasterisationInput(
                                    point_cloud=self.scene.point_cloud,
                                    point_cloud_features=self.scene.point_cloud_features,
                                    point_invalid_mask=self.scene.point_invalid_mask,
                                    point_object_id=self.scene.point_object_id,
                                    camera_info=resized_camera_info,
                                    current_pose=current_pose_tensor,
                                    color_max_sh_band=3,
                                )
                            )

                            predicted_image = torch.clamp(
                                predicted_image, min=0, max=1)
                            predicted_image = predicted_image.permute(2, 0, 1)

                            predicted_depth = predicted_depth.cuda()
                            predicted_depth = predicted_depth / \
                                torch.max(predicted_depth)
                            ground_truth_image = ground_truth_image_torch_list[v]
                            if len(predicted_image.shape) == 3:
                                predicted_image_temp = predicted_image.unsqueeze(0)
                            if len(ground_truth_image.shape) == 3:
                                ground_truth_image_temp = ground_truth_image.unsqueeze(
                                    0)

                            # DEBUG =======================
                            depth_map = depth_image_torch_list[v]
                            depth_mask = torch.where(depth_map >= 0, True, False)
                            depth_map = depth_map / torch.max(depth_map)
                            depth_map = depth_map.squeeze(0)
                            # =============================

                            # sum over all images
                            L1 += 0.8*torch.abs(predicted_image_temp - ground_truth_image_temp).mean() + 0.2*(1 - ssim(predicted_image_temp, ground_truth_image_temp,
                                                                                                                    data_range=1, size_average=True))

                            # Depth backprop
                            masked_difference = torch.abs(
                                predicted_depth - depth_map)  # [depth_mask]
                            L_DEPTH = masked_difference.mean()
                            if len(masked_difference) == 0:
                                L_DEPTH = torch.tensor(0)

                            L = L1 + lambda_depth * L_DEPTH # 


                            L.backward(retain_graph=True)   #
                            # gradient = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                            #                                 pypose_bspline_knots],grad_outputs=current_pose_tensor.grad)#, retain_graph=True)#torch.ones_like(current_pose_tensor)

                            # dL1_d_pypose_knots = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                            #                                pypose_bspline_knots],grad_outputs=current_pose_tensor.grad, retain_graph=True)#torch.ones_like(current_pose_tensor)
                            # pypose_bspline_knots.grad = dL1_d_pypose_knots[0]

                            # DEBUG
                            # if start_t_optimization:
                            dL1_d_pypose_knots_t = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                                pypose_bspline_knots_t], grad_outputs=current_pose_tensor.grad, retain_graph=True)
                            # current_pose_tensor.grad.detach_()
                            # else:
                            dL1_d_pypose_knots_q = torch.autograd.grad(outputs=[current_pose_tensor], inputs=[
                                pypose_bspline_knots_q], grad_outputs=current_pose_tensor.grad, retain_graph=True)

                            if (dL1_d_pypose_knots_t) is not None:
                                # if (gradient) is not None:
                                # optimizer_bspline_bases.zero_grad()
                                # optimizer_bspline_bases_t.zero_grad()
                                # optimizer_bspline_bases_q.zero_grad()

                                pypose_bspline_knots_t.grad = dL1_d_pypose_knots_t[0]

                                if (not torch.isnan(pypose_bspline_knots_t.grad).any()):
                                    if (not torch.isinf(pypose_bspline_knots_t.grad).any()):
                                        optimizer_bspline_bases_t.step()
                                        optimizer_bspline_bases_t.zero_grad()

                            if (dL1_d_pypose_knots_q) is not None:
                                pypose_bspline_knots_q.grad = dL1_d_pypose_knots_q[0]
                                if (not torch.isnan(pypose_bspline_knots_q.grad).any()):
                                    if (not torch.isinf(pypose_bspline_knots_q.grad).any()):
                                        optimizer_bspline_bases_q.step()
                                        optimizer_bspline_bases_q.zero_grad()

                        if epoch % 5 == 0 : #30 for 4 bases
                            scheduler_t.step()

                            for param_group in optimizer_bspline_bases_t.param_groups:
                                if param_group['lr'] < 1e-5: 
                                    param_group['lr'] = 1e-5

                        if epoch % 5 == 0:  # 5
                            scheduler_q.step()
                            for param_group in optimizer_bspline_bases_q.param_groups:
                                if param_group['lr'] < 1e-5:
                                    param_group['lr'] = 1e-5
                                    
                        if epoch > 0 and not start_t_optimization:
                            start_t_optimization = True
                            print("Setting start_t_optimization to True")

                        if epoch % 100 == 0:
                            print(f"Current photometric loss: {L1}")
                            print(f"Current  loss: {L}")

                            # DEBUG visualization ===========================
                            temp_data = []

                            # Iterate over groundtruth
                            for p, pose in enumerate(groundtruth_pose):
                                position_data = np.array(
                                    pose.position().data[:]).reshape(1, 3)
                                rotation_data = np.array(
                                    pose.rotation().data[:]).reshape(1, 4)
                                temp_data.append(
                                    np.hstack([position_data, rotation_data]))

                            f1 = pl.figure(1)
                            a3d = f1.add_subplot(111, projection='3d')

                            f2 = pl.figure(2)
                            a2d = f2.add_subplot(111)

                            f3 = pl.figure(3)
                            a2d_bases = f3.add_subplot(111)
                            f4 = pl.figure(4)
                            a2d_bases_t = f4.add_subplot(111)

                            a3d.set_xlabel('X')
                            a3d.set_ylabel('Y')
                            a3d.set_zlabel('Z')

                            # Plot current trajectory guess
                            with torch.no_grad():
                                pypose_bspline_knots = torch.hstack(
                                    (pypose_bspline_knots_t, pypose_bspline_knots_q))
                                plotCoordinateFrame.plot_trajectory(
                                    a3d, pypose_bspline_knots[:, :3], color="black", linewidth=1, label="estimation", evaluate_zspline=True)
                                plotCoordinateFrame.plot_trajectory_2d(
                                    a2d, pypose_bspline_knots[:, :3], color="black", linewidth=1, label="estimation", evaluate_zspline=True)

                                for k in range(4):
                                    a2d_bases.plot(error_bases_q[k, :epoch])
                                    a2d_bases_t.plot(error_bases_t[k, :epoch])

                                # Plot pose error
                                t_step = 1/batch_size
                                t_range = t_step*np.arange(0, batch_size)
                                poses = curve_evaluation.cubic_bspline_interpolation(
                                    pp.SE3(pypose_bspline_knots.double()), torch.tensor(t_range).double(), enable_eps=False, enable_z_spline=True)
                                gt_poses = curve_evaluation.cubic_bspline_interpolation(pp.SE3(
                                    pypose_groundtruth_bspline_knots.double()), torch.tensor(t_range).double(), enable_eps=True, enable_z_spline=True)

                                # poses = curve_evaluation.bspline(
                                #     pp.SE3(pypose_bspline_knots.unsqueeze(0)), torch.tensor(t_range, device="cpu"),enable_z_spline=True).squeeze(0)
                                # gt_poses = curve_evaluation.bspline(
                                #     pp.SE3(pypose_groundtruth_bspline_knots.unsqueeze(0)), torch.tensor(t_range, device="cpu"),enable_z_spline=True).squeeze(0)

                                for pose in poses:
                                    plotCoordinateFrame.plotCoordinateFrame(
                                        a2d, pp.matrix(pose), size=0.5, linewidth=0.5)
                                    plotCoordinateFrame.plotCoordinateFrame(
                                        a3d, pp.matrix(pose), size=0.5, linewidth=0.5)

                                a2d.scatter(
                                    poses[:, 0], poses[:, 1], s=5, color="black")
                                a3d.scatter(poses[:, 0], poses[:, 1],
                                            poses[:, 2], s=5, color="black")

                                # Add scatter for reconstructed bases
                                bspline_bases_numpy = pypose_bspline_knots.clone().detach().cpu().numpy()
                                for bspline_base in pypose_bspline_knots:
                                    plotCoordinateFrame.plotCoordinateFrame(a2d, pp.matrix(
                                        pp.SE3(bspline_base)), size=0.5, linewidth=0.5)
                                    plotCoordinateFrame.plotCoordinateFrame(a3d, pp.matrix(
                                        pp.SE3(bspline_base)), size=0.5, linewidth=0.5)

                                a2d.scatter(
                                    bspline_bases_numpy[:, 0], bspline_bases_numpy[:, 1], s=5, color="gray")
                                a3d.scatter(
                                    bspline_bases_numpy[:, 0], bspline_bases_numpy[:, 1], bspline_bases_numpy[:, 2], s=5, color="gray")

                            # Plot groundtruth trajectory
                            # plotCoordinateFrame.plot_trajectory_lie(
                            #     a3d, groundtruth_bases, linewidth=1, resolution=0.1, size=0.2)
                            plotCoordinateFrame.plot_trajectory(
                                a3d, groundtruth_bases[:, 3:], color="orange", linewidth=1, label="groundtruth", evaluate_zspline=True)
                            plotCoordinateFrame.plot_trajectory_2d(
                                a2d, groundtruth_bases[:, 3:], color="orange", linewidth=1, label="groundtruth", evaluate_zspline=True)

                            groundtruth_delta = np.array(
                                [groundtruth_pose[k].to_tangent(_EPS) for k in range(batch_size)])

                            # Add scatter for groundtruth points
                            a2d.scatter(gt_poses[:, 0], gt_poses[:, 1],
                                        s=5, color="orange", label="Interpolated Groundtruth discrete poses")
                            a3d.scatter(gt_poses[:, 0], gt_poses[:, 1],
                                        gt_poses[:, 2], s=5, color="orange", label="Interpolated Groundtruth discrete poses")
                            a2d.scatter(groundtruth_delta[:, 3], groundtruth_delta[:, 4],
                                        s=5, color="yellow", label="Groundtruth discrete poses")
                            a3d.scatter(groundtruth_delta[:, 3], groundtruth_delta[:, 4],
                                        groundtruth_delta[:, 5], s=5, color="yellow", label="Groundtruth discrete poses")
                            for k in range(batch_size):
                                plotCoordinateFrame.plotCoordinateFrame(
                                    a2d, groundtruth_pose[k].to_homogenous_matrix(), size=0.5, linewidth=1)
                                plotCoordinateFrame.plotCoordinateFrame(
                                    a3d, groundtruth_pose[k].to_homogenous_matrix(), size=0.5, linewidth=1)

                            # Add scatter for groundtruth bases
                            a2d.scatter(pypose_groundtruth_bspline_knots[:, 0], pypose_groundtruth_bspline_knots[:, 1],
                                        s=5, color="red", label="Groundtruth spline bases")
                            a3d.scatter(pypose_groundtruth_bspline_knots[:, 0], pypose_groundtruth_bspline_knots[:, 1],
                                        pypose_groundtruth_bspline_knots[:, 2], s=5, color="red", label="Groundtruth spline bases")

                            a3d.legend()
                            a2d.legend()

                            a2d.axis("equal")
                            a3d.axis("equal")

                            a2d.set_xlim([-1.5, 1])
                            a2d.set_ylim([-1.5, 1])
                            a3d.set_xlim([-2.5, 2])
                            a3d.set_ylim([-2, 2.5])
                            a3d.set_zlim([-1, 1])

                            a2d.set_aspect('equal', adjustable='box')
                            a2d.grid()

                            f1.savefig(os.path.join(
                                self.output_path, f"3d_plots_batch_{i}", f'figure_{epoch}.png'))
                            f2.savefig(os.path.join(
                                self.output_path, f"2d_plots_batch_{i}", f'figure_{epoch}_2d.png'))
                            f3.savefig(os.path.join(
                                self.output_path, f"2d_plots_batch_{i}", f'Error_bases_q.png'))
                            f4.savefig(os.path.join(
                                self.output_path, f"2d_plots_batch_{i}", f'Error_bases_t.png'))
                            pickle.dump(f1, open(os.path.join(
                                self.output_path, f"pickle_files_batch_{i}", f'FigureObject.fig_{epoch}.pickle'), 'wb'))
                            f1.clear()
                            f2.clear()
                            f3.clear()
                            f4.clear()

                            mean_error_t = np.mean(errors_t, axis=0)
                            for k in range(batch_size):
                                plt.plot(
                                    errors_t[k, :epoch], label=f"Error frame {k}", linewidth=0.5)
                                plt.xlabel("Epoch")
                                plt.ylabel("Error")
                                plt.title("Translational error")
                                plt.legend()
                            plt.plot(mean_error_t[:epoch],
                                    label=f"Mean error", linewidth=1)
                            plt.savefig(
                                os.path.join(self.output_path, f"trasl_error_segment{i}.png"))
                            plt.clf()

                            mean_error_q = np.mean(errors_q, axis=0)
                            for k in range(batch_size):
                                plt.plot(
                                    errors_q[k, :epoch], label=f"Error frame {k}", linewidth=0.5)
                                plt.xlabel("Epoch")
                                plt.ylabel("Error")
                                plt.title("Rotational error")
                                plt.legend()
                            plt.plot(mean_error_q[:epoch],
                                    label=f"Mean error", linewidth=1)
                            plt.savefig(
                                os.path.join(self.output_path, f"rot_error_segment{i}.png"))
                            plt.clf()

                            # ============================================
                    bases[i:i+4, :] = pypose_bspline_knots.clone().detach()
                    torch.save(bases, os.path.join(self.output_path, f"bases.pt"))


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
