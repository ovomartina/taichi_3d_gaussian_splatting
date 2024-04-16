# %%
from .GaussianPointCloudScene import GaussianPointCloudScene
from .ImagePoseDataset import ImagePoseDataset
from .Camera import CameraInfo
from .GaussianPointCloudPoseRasterisation import GaussianPointCloudPoseRasterisation
from .GaussianPointAdaptiveController import GaussianPointAdaptiveController
from .LossFunction import LossFunction
from .Lidar import Lidar
from .utils import quaternion_to_rotation_matrix_torch, inverse_SE3, SE3_to_quaternion_and_translation_torch

import torch
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from pytorch_msssim import ssim
from tqdm import tqdm
import taichi as ti
import os
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, Tuple
import numpy as np
import io
import PIL.Image
from torchvision.transforms import ToTensor
import sym
import pandas as pd

# DEBUG - allow reproducibility
torch.manual_seed(42)

_EPS = 1e-6  # Possible problems with 1e-8 because of using float32
def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def collate_fn(batch):
    # Sort the batch based on the "index" field
    sorted_batch = sorted(batch, key=lambda x: x['index'])
    return sorted_batch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


def merge_scenes(scene_list):
    # the config does not matter here, only for training

    merged_point_cloud = torch.cat(
        [scene.point_cloud for scene in scene_list], dim=0)
    merged_point_cloud_features = torch.cat(
        [scene.point_cloud_features for scene in scene_list], dim=0)
    num_of_points_list = [scene.point_cloud.shape[0]
                          for scene in scene_list]
    start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
    end_offset_list = np.cumsum(num_of_points_list).tolist()
    # self.extra_scene_info_dict = {
    #     idx: self.ExtraSceneInfo(
    #         start_offset=start_offset,
    #         end_offset=end_offset,
    #         center=scene_list[idx].point_cloud.mean(dim=0),
    #         visible=True
    #     ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
    # }
    point_object_id = torch.zeros(
        (merged_point_cloud.shape[0],), dtype=torch.int32, device="cuda")
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


class GaussianPointTrainerBundleAdjustment:
    delta_list = []
    delta_q_list = []
    delta_t_list = []

    # optimizers_list = []
    optimizers_q_list = []
    optimizers_t_list = []

    pose_scheduler_list = []
    q_scheduler_list = []
    t_scheduler_list = []
    errors_q = []
    errors_t = []

    iterations_pose = 1
    iterations_bundle_adjustment = 5
    iterations_pose_factor = 1.5
    iterations_bundle_adjustment_factor = 1.5

    @dataclass
    class TrainConfig(YAMLWizard):
        train_dataset_json_path: str = ""
        val_dataset_json_path: str = ""
        pointcloud_parquet_path: str = ""
        num_iterations: int = 300000
        val_interval: int = 1000
        feature_learning_rate: float = 1e-3
        position_learning_rate: float = 1e-5
        position_learning_rate_decay_rate: float = 0.97
        position_learning_rate_decay_interval: int = 100
        increase_color_max_sh_band_interval: int = 1000.  # 1000
        log_loss_interval: int = 10
        log_metrics_interval: int = 100
        print_metrics_to_console: bool = False
        log_image_interval: int = 1000
        enable_taichi_kernel_profiler: bool = False
        log_taichi_kernel_profile_interval: int = 1000
        log_validation_image: bool = True
        initial_downsample_factor: int = 4
        half_downsample_factor_interval: int = 25  # 250
        summary_writer_log_dir: str = "logs"
        output_model_dir: Optional[str] = None
        rasterisation_config: GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationConfig = GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationConfig()
        adaptive_controller_config: GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig()
        gaussian_point_cloud_scene_config: GaussianPointCloudScene.PointCloudSceneConfig = GaussianPointCloudScene.PointCloudSceneConfig()
        loss_function_config: LossFunction.LossFunctionConfig = LossFunction.LossFunctionConfig()
        noise_std_q: float = 0.0
        noise_std_t: float = 0.0
        camera_pose_learning_rate_decay_rate: float = 0.97
        # increase_ba_iterations_interval: int = 10000
        start_pose_optimization: int = 7000
        save_position: int = 2000

    def __init__(self, config: TrainConfig):
        self.config = config
        # create the log directory if it doesn't exist
        os.makedirs(self.config.summary_writer_log_dir, exist_ok=True)

        if self.config.output_model_dir is None:
            self.config.output_model_dir = self.config.summary_writer_log_dir
            os.makedirs(self.config.output_model_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=self.config.summary_writer_log_dir)

        self.train_dataset = ImagePoseDataset(
            dataset_json_path=self.config.train_dataset_json_path,
            noise_std_q=self.config.noise_std_q,
            noise_std_t=self.config.noise_std_t
        )
        self.val_dataset = ImagePoseDataset(
            dataset_json_path=self.config.val_dataset_json_path,
            noise_std_q=self.config.noise_std_q,
            noise_std_t=self.config.noise_std_t
        )

        for i in range(len(self.train_dataset)):
            # self.delta_list.append(torch.zeros((6, 1)))
            # self.delta_list[i] = self.delta_list[i].cuda()
            # self.delta_list[i].requires_grad = True

            self.delta_q_list.append(torch.zeros((3, 1), dtype=torch.float64))
            self.delta_q_list[i] = self.delta_q_list[i].cuda()
            self.delta_q_list[i].requires_grad = True

            self.delta_t_list.append(torch.zeros((3, 1), dtype=torch.float64))
            self.delta_t_list[i] = self.delta_t_list[i].cuda()
            self.delta_t_list[i].requires_grad = True

            # Separate optimization for q and t
            self.optimizers_q_list.append(torch.optim.Adam(
                [self.delta_q_list[i]], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2))
            self.optimizers_t_list.append(torch.optim.Adam(
                [self.delta_t_list[i]], lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-2))

            self.q_scheduler_list.append(torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizers_q_list[i], gamma=0.9947))
            self.t_scheduler_list.append(torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizers_t_list[i], gamma=0.9947))

        transform = np.array([[-0.07269247,  0.12011554, -0.46715833, -1.61574687],
                              [0.48165509,  0.04348471, -0.06376747,  0.4119509],
                              [0.02594256, -0.47077614, -0.12508256, -0.20647023],
                              [0.,          0.,          0.,          1.]])
        transform = np.eye(4)
        self.scene = GaussianPointCloudScene.from_parquet(
            self.config.pointcloud_parquet_path, config=self.config.gaussian_point_cloud_scene_config, transform=transform)

        # # DEBUG: load scene already trained 30k iterations
        # scene = GaussianPointCloudScene.from_parquet(
        #     "logs/replica_colmap/room_1_high_quality_500_frames_noisy_lidar/scene_30000.parquet", config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        # self.scene = merge_scenes([scene])

        self.scene = self.scene.cuda()
        self.scene.point_cloud = self.scene.point_cloud.contiguous()

        self.adaptive_controller = GaussianPointAdaptiveController(
            config=self.config.adaptive_controller_config,
            maintained_parameters=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerMaintainedParameters(
                pointcloud=self.scene.point_cloud,
                pointcloud_features=self.scene.point_cloud_features,
                point_invalid_mask=self.scene.point_invalid_mask,
                point_object_id=self.scene.point_object_id,
            ))
        self.rasterisation = GaussianPointCloudPoseRasterisation(
            config=self.config.rasterisation_config,
            backward_valid_point_hook=self.adaptive_controller.update,
        )

        self.loss_function = LossFunction(
            config=self.config.loss_function_config)

        self.best_psnr_score = 0.

    @staticmethod
    def _downsample_image_and_camera_info(image: torch.Tensor, depth: torch.Tensor, camera_info: CameraInfo, depth_map: torch.Tensor, downsample_factor: int):
        camera_height = camera_info.camera_height // downsample_factor
        camera_width = camera_info.camera_width // downsample_factor
        image = transforms.functional.resize(image, size=(
            camera_height, camera_width), antialias=True)
        if depth_map is not None:
            depth_map = transforms.functional.resize(depth_map, size=(
                camera_height, camera_width), antialias=True)
        if depth is not None:
            depth = transforms.functional.resize(depth, size=(
                camera_height, camera_width), antialias=True)
        camera_width = camera_width - camera_width % 16
        camera_height = camera_height - camera_height % 16
        image = image[:3, :camera_height, :camera_width].contiguous()
        if depth_map is not None:
            depth_map = depth_map[:3, :camera_height,
                                  :camera_width].contiguous()
        if depth is not None:
            depth = depth[:3, :camera_height, :camera_width].contiguous()
        camera_intrinsics = camera_info.camera_intrinsics
        camera_intrinsics = camera_intrinsics.clone()
        camera_intrinsics[0, 0] /= downsample_factor
        camera_intrinsics[1, 1] /= downsample_factor
        camera_intrinsics[0, 2] /= downsample_factor
        camera_intrinsics[1, 2] /= downsample_factor
        resized_camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_id=camera_info.camera_id)
        return image, resized_camera_info, depth, depth_map

    def train(self):
        # we don't use taichi fields, so we don't need to allocate memory, but taichi requires the memory to be allocated > 0
        ti.init(arch=ti.cuda, device_memory_GB=0.1,
                kernel_profiler=self.config.enable_taichi_kernel_profiler)
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=4)
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)
        train_data_loader_iter = cycle(train_data_loader)

        optimizer = torch.optim.Adam(
            [self.scene.point_cloud_features], lr=self.config.feature_learning_rate, betas=(0.9, 0.999))
        position_optimizer = torch.optim.Adam(
            [self.scene.point_cloud], lr=self.config.position_learning_rate, betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=position_optimizer, gamma=self.config.position_learning_rate_decay_rate)
        downsample_factor = self.config.initial_downsample_factor

        recent_losses = deque(maxlen=100)

        previous_problematic_iteration = -1000

        # If it's the first time loading index, load from json. Otherwise use last estimation
        first_iteration = [True for _ in range(len(self.train_dataset))]
        initial_noisy_poses = [[] for _ in range(len(self.train_dataset))]
        initial_q = [np.array([0., 0., 0., 1.])
                     for _ in range(len(self.train_dataset))]
        initial_t = [np.array([0., 0., 0.])
                     for _ in range(len(self.train_dataset))]
        groundtruth_q = [[] for _ in range(len(self.train_dataset))]
        groundtruth_t = [[] for _ in range(len(self.train_dataset))]
        self.errors_q = [[] for _ in range(len(self.train_dataset))]
        self.errors_t = [[] for _ in range(len(self.train_dataset))]
        pose_iterations_count = [0 for _ in range(len(self.train_dataset))]
        optimal_delta = [np.zeros((1, 6))
                         for _ in range(len(self.train_dataset))]
        total_iteration_count = 0
        scene_iteration_count = 0
        pose_iteration_count = 0
        optimize_pose = False
        previous_smooth = None
        previous_depth_loss = None
        previous_ssim_loss = None
        increase_pose_iterations_from = self.config.start_pose_optimization
        count_10 = 0
        df = pd.read_json(
            "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/data/replica_colmap/room_1_high_quality_500_frames/train_groundtruth.json", orient="records")

        while total_iteration_count < self.config.num_iterations:
            image_gt, q_pointcloud_camera_current, t_pointcloud_camera_current, camera_info, depth_gt, lidar_pcd, t_lidar_camera, T_pointcloud_camera, index, q_pointcloud_camera_gt, t_pointcloud_camera_gt = next(
                train_data_loader_iter)

            # # DEBUG colmap =======================================================
            # q_pointcloud_camera_current = q_pointcloud_camera_gt
            # t_pointcloud_camera_current = t_pointcloud_camera_gt
            # groundtruth_T_colmap_camera = torch.tensor(
            #     df.iloc[index]["T_colmap_camera"])
            # q_pointcloud_camera_gt, t_pointcloud_camera_gt = SE3_to_quaternion_and_translation_torch(groundtruth_T_colmap_camera.unsqueeze(0))
            # # ====================================================================
            
            # Sanity check
            # if total_iteration_count < self.config.start_pose_optimization:
            #     groundtruth_q[index] = q_pointcloud_camera_gt
            #     groundtruth_t[index] = t_pointcloud_camera_gt

            #     q_pointcloud_camera, t_pointcloud_camera = q_pointcloud_camera_gt, t_pointcloud_camera_gt

            #     # Generate Pose3 Object
            #     q_pointcloud_camera_numpy = q_pointcloud_camera.detach().cpu().numpy()
            #     t_pointcloud_camera_numpy = t_pointcloud_camera.detach().cpu().numpy()
            #     noisy_rotation = sym.Rot3(
            #         q_pointcloud_camera_numpy[0, :])
            #     initial_noisy_pose = sym.Pose3(
            #         R=noisy_rotation, t=t_pointcloud_camera_numpy.T.astype("float"))
            #     initial_noisy_poses[index] = initial_noisy_pose

            #     initial_q[index] = q_pointcloud_camera_numpy
            #     initial_t[index] = t_pointcloud_camera_numpy

            # else:
            delta_tensor = torch.cat(
                (self.delta_q_list[index], self.delta_t_list[index]), axis=0).contiguous()

            if first_iteration[index]:
                groundtruth_q[index] = q_pointcloud_camera_gt
                groundtruth_t[index] = t_pointcloud_camera_gt

                q_pointcloud_camera, t_pointcloud_camera = q_pointcloud_camera_current, t_pointcloud_camera_current

                first_iteration[index] = False

                # Generate Pose3 Object
                noisy_q_pointcloud_camera_numpy = q_pointcloud_camera.detach().cpu().numpy()
                noisy_t_pointcloud_camera_numpy = t_pointcloud_camera.detach().cpu().numpy()
                noisy_rotation = sym.Rot3(
                    noisy_q_pointcloud_camera_numpy[0, :])
                initial_noisy_pose = sym.Pose3(
                    R=noisy_rotation, t=noisy_t_pointcloud_camera_numpy.T.astype("float"))
                initial_noisy_poses[index] = initial_noisy_pose

                groundtruth_rotation = sym.Rot3(
                    groundtruth_q[index].detach().cpu().numpy().reshape((1, 4)))
                groundtruth_pose = sym.Pose3(
                    R=groundtruth_rotation, t=groundtruth_t[index].detach().cpu().numpy().reshape((1, 3)).astype("float"))

                # Save optimal transformation for later
                delta_pose_2 = sym.Pose3.local_coordinates(
                    initial_noisy_poses[index], groundtruth_pose)  # THIS IS CORRECT
                optimal_delta[index] = (delta_pose_2).reshape((1, 6))
                debug = sym.Pose3.retract(
                    initial_noisy_poses[index], optimal_delta[index].reshape((6, 1)),  0.0000001)  # THIS IS CORRECT
                print(
                    f"Debug: rotation {debug.rotation()}, translation: {debug.position()}")
                # print(f"Storage D tangent: {debug.storage_D_tangent()}")
                # initial_q[index] = q_pointcloud_camera_current.cpu().numpy()
                # initial_t[index] = t_pointcloud_camera_current.cpu().numpy()

                # set to GT
                initial_q[index] = noisy_q_pointcloud_camera_numpy
                initial_t[index] = noisy_t_pointcloud_camera_numpy
            else:
                # delta_numpy_array = self.delta_list[index].clone(
                # ).detach().cpu().numpy()
                delta_numpy_array = delta_tensor.clone(
                ).detach().cpu().numpy()
                current_pose = sym.Pose3.retract(
                    initial_noisy_poses[index], delta_numpy_array, _EPS)
                q_pointcloud_camera, t_pointcloud_camera = extract_q_t_from_pose(
                    initial_noisy_poses[index])

            image_gt = image_gt.cuda()
            depth_gt = depth_gt.cuda()
            q_pointcloud_camera = q_pointcloud_camera.cuda()
            t_pointcloud_camera = t_pointcloud_camera.cuda()
            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
            camera_info.camera_width = int(camera_info.camera_width)
            camera_info.camera_height = int(camera_info.camera_height)
            error_q = 0.
            error_t = 0.
            plt.figure()
            image_gt_original = image_gt
            camera_info_original = camera_info
            depth_gt_original = depth_gt

            # if total_iteration_count%increase_pose_iterations==0:
            #         pose_iterations = int(pose_iterations*pose_iteration_increase_factor)
            pose_iterations = 10
            coarse_iterations = 0
            # if total_iteration_count > increase_pose_iterations_from:
            #     pose_iterations = 30

            if optimize_pose:
                print("optimizing pose")
                # for k in range(30):
                for k in range(pose_iterations):

                    image_gt = image_gt_original
                    camera_info = camera_info_original
                    depth_gt = depth_gt_original
                    # if total_iteration_count % self.config.half_downsample_factor_interval == 0 and total_iteration_count > 0 and downsample_factor > 1:
                    #     downsample_factor = downsample_factor // 2
                    # if downsample_factor > 1:
                    #     image_gt, camera_info, depth_gt, _ = GaussianPointTrainerBundleAdjustment._downsample_image_and_camera_info(
                    #         image_gt, depth_gt, camera_info, None, downsample_factor=downsample_factor)
                    # else:
                    #     image_gt, camera_info, depth_gt, _ = GaussianPointTrainerBundleAdjustment._downsample_image_and_camera_info(
                    #         image_gt, depth_gt, camera_info, None, 2)  # Always downsample image for pose optimization

                    if k < coarse_iterations and downsample_factor < 4:
                        coarse_downsample_factor = downsample_factor * 2
                        try:
                            assert downsample_factor <= 4
                        except Exception as e:
                            print("Coarse downsample factor")
                            print(e)
                        image_gt, camera_info, depth_gt, _ = GaussianPointTrainerBundleAdjustment._downsample_image_and_camera_info(
                            image_gt, depth_gt, camera_info, None, downsample_factor=coarse_downsample_factor)  # downsample_factor=1

                    for i in range(len(self.train_dataset)):
                        self.optimizers_q_list[i].zero_grad()
                        self.optimizers_t_list[i].zero_grad()

                    delta_tensor = torch.cat(
                        (self.delta_q_list[index], self.delta_t_list[index]), axis=0).contiguous()

                    with torch.no_grad():
                        # Save for plotting
                        delta_numpy_array = delta_tensor.clone(
                        ).detach().cpu().numpy()
                        initial_rotation = sym.Rot3(
                            initial_q[index][0, :])
                        initial_pose = sym.Pose3(
                            R=initial_rotation, t=initial_t[index].astype("float"))
                        current_pose = sym.Pose3.retract(
                            initial_pose, delta_numpy_array, _EPS)
                        current_q, current_t = extract_q_t_from_pose(
                            current_pose)

                        # Compute angle error in radians
                        q_pointcloud_camera_gt_inverse = q_pointcloud_camera_gt * \
                            np.array([-1., -1., -1., 1.])
                        q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
                            (1, 4)), current_q.reshape((1, 4)),)
                        q_difference = q_difference.cpu().numpy()
                        angle_difference = np.abs(
                            2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))

                        if angle_difference > np.pi:
                            angle_difference = 2*np.pi - angle_difference
                        error_q = angle_difference
                        error_t = torch.linalg.vector_norm(
                            current_t - t_pointcloud_camera_gt)

                        if not np.isnan(error_q):
                            self.errors_q[index].append(error_q)
                        else:
                            print(f"Current delta:\n\t{delta_numpy_array}")
                            print(f"Current pose:\n\t{current_pose}")
                            print("Angle error_q is Nan")
                            print(f"q_pointcloud_camera_gt: \n\t{q_pointcloud_camera_gt}, \
                                  q_pointcloud_camera_gt_inverse: \n\t {q_pointcloud_camera_gt_inverse} \
                                  q_difference: \n\t {q_difference}, \
                                  current_q: {current_q}, \
                                  angle_difference: {angle_difference}")
                            
                        self.errors_t[index].append(error_t)

                    gaussian_point_cloud_rasterisation_input = GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info,
                        delta=delta_tensor,
                        initial_q=initial_q[index],
                        initial_t=initial_t[index],
                        color_max_sh_band=3,
                    )

                    image_pred, image_depth, pixel_valid_point_count, _ = self.rasterisation(
                        gaussian_point_cloud_rasterisation_input
                    )

                    # clip to [0, 1]
                    image_pred = torch.clamp(image_pred, min=0, max=1)
                    # hxwx3->3xhxw
                    image_pred = image_pred.permute(2, 0, 1)

                    if len(image_pred.shape) == 3:
                        image_pred = image_pred.unsqueeze(0)
                    if len(image_gt.shape) == 3:
                        image_gt = image_gt.unsqueeze(0)
                    print(f"Image predicted shape:{image_pred.shape}")
                    print(f"Image gt shape:{image_gt.shape}")
                    loss = 0.8*torch.abs(image_pred - image_gt).mean() + 0.2*(1 - ssim(image_pred, image_gt,
                                                                                       data_range=1, size_average=True))

                    loss.backward()

                    if (not torch.isnan(self.delta_q_list[index].grad).any()):
                        self.optimizers_q_list[index].step()
                    if (not torch.isnan(self.delta_t_list[index].grad).any()):
                        self.optimizers_t_list[index].step()
                    if pose_iterations_count[index] % 10 == 0:
                        self.q_scheduler_list[index].step()
                        self.t_scheduler_list[index].step()

                    # Set minimum learning rate (from BAD gaussians)
                    if get_lr(self.optimizers_q_list[index]) < 1e-5:
                        for g in self.optimizers_q_list[index].param_groups:
                            g['lr'] = 1e-5

                    if get_lr(self.optimizers_t_list[index]) < 1e-5:
                        for g in self.optimizers_t_list[index].param_groups:
                            g['lr'] = 1e-5

                    magnitude_grad_viewspace_on_image = None
                    if self.adaptive_controller.input_data is not None:
                        magnitude_grad_viewspace_on_image = self.adaptive_controller.input_data.magnitude_grad_viewspace_on_image
                        self._plot_grad_histogram(
                            self.adaptive_controller.input_data, writer=self.writer, iteration=total_iteration_count)
                        self._plot_value_histogram(
                            self.scene, writer=self.writer, iteration=total_iteration_count)
                        self.writer.add_histogram(
                            "train/pixel_valid_point_count", pixel_valid_point_count, total_iteration_count)

                    if total_iteration_count % self.config.log_loss_interval == 0:
                        self.writer.add_scalar(
                            "train/loss", loss.item(), total_iteration_count)

                        if previous_smooth is not None and previous_depth_loss is not None and previous_ssim_loss is not None:
                            self.writer.add_scalar(
                                "train/smooth_loss", previous_smooth, total_iteration_count)
                            self.writer.add_scalar(
                                "train/depth loss", previous_depth_loss, total_iteration_count)
                            self.writer.add_scalar(
                                "train/ssim loss", previous_ssim_loss, total_iteration_count)

                        if optimize_pose:
                            last_q_estimate = np.array([np.array(arr)[-1]
                                                        for arr in self.errors_q if len(arr) > 0])
                            mean_last_q_estimate = np.mean(
                                last_q_estimate) if last_q_estimate.size != 0 else 0
                            self.writer.add_scalar(
                                "train/total_error_q", mean_last_q_estimate, total_iteration_count)
                            last_t_estimate = np.array([np.array(arr)[-1]
                                                        for arr in self.errors_t if len(arr) > 0])
                            mean_last_t_estimate = np.mean(
                                last_t_estimate) if last_t_estimate.size != 0 else 0
                            self.writer.add_scalar(
                                "train/total_error_t", mean_last_t_estimate, total_iteration_count)

                        if self.config.print_metrics_to_console:
                            print(f"train_iteration={total_iteration_count};")
                            print(f"train_loss={loss.item()};")
                            print(f"train_l1_loss={l1_loss.item()};")
                            print(f"train_ssim_loss={ssim_loss.item()};")

                    if self.config.enable_taichi_kernel_profiler and total_iteration_count % self.config.log_taichi_kernel_profile_interval == 0 and total_iteration_count > 0:
                        ti.profiler.print_kernel_profiler_info("count")
                        ti.profiler.clear_kernel_profiler_info()
                    if total_iteration_count % self.config.log_metrics_interval == 0:
                        psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                            image_pred=image_pred, image_gt=image_gt)
                        self.writer.add_scalar(
                            "train/psnr", psnr_score.item(), total_iteration_count)
                        self.writer.add_scalar(
                            "train/ssim", ssim_score.item(), total_iteration_count)
                        if self.config.print_metrics_to_console:
                            print(f"train_psnr={psnr_score.item()};")
                            print(
                                f"train_psnr_{total_iteration_count}={psnr_score.item()};")
                            print(f"train_ssim={ssim_score.item()};")
                            print(
                                f"train_ssim_{total_iteration_count}={ssim_score.item()};")

                    is_problematic = False
                    if len(recent_losses) == recent_losses.maxlen and total_iteration_count - previous_problematic_iteration > recent_losses.maxlen:
                        avg_loss = sum(recent_losses) / len(recent_losses)
                        if loss.item() > avg_loss * 1.5:
                            is_problematic = True
                            previous_problematic_iteration = total_iteration_count

                    if total_iteration_count % self.config.save_position == 0:
                        for i in range(len(self.errors_q)):
                            # Save position
                            # Accessing the first entry of the list
                            entry_q = self.errors_q[i]
                            numpy_array_q = np.array(
                                [tensor.item() for tensor in entry_q])
                            np.savetxt(
                                f'errors/error_q_{i}.txt', numpy_array_q)

                            # Accessing the first entry of the list
                            entry_t = self.errors_t[i]
                            numpy_array_t = np.array(
                                [tensor.item() for tensor in entry_t])
                            np.savetxt(
                                f'errors/error_t_{i}.txt', numpy_array_t)

                    del image_pred, image_depth, loss
                    # they use 7000 in paper, it's hard to set a interval so hard code it here
                    if (total_iteration_count % self.config.val_interval == 0 and total_iteration_count != 0) or total_iteration_count == 7000 or total_iteration_count == 5000:
                        self.validation(
                            val_data_loader, total_iteration_count, initial_q, initial_t)

                    total_iteration_count += 1
                    pose_iterations_count[index] += 1

                delta_tensor = torch.cat(
                    (self.delta_q_list[index], self.delta_t_list[index]), axis=0).contiguous()
                delta_numpy_array = delta_tensor.clone(
                ).detach().cpu().numpy()
                current_pose_estimate = sym.Pose3.retract(
                    initial_noisy_poses[index], delta_numpy_array, _EPS)
                current_q_estimate, current_t_estimate = extract_q_t_from_pose(
                    current_pose_estimate)
                try:
                    with torch.no_grad():
                        error_q_numpy = np.array(
                            self.errors_q[index]).T   # Epochs x 1
                        error_q_numpy = np.expand_dims(error_q_numpy, axis=0)
                        plt.plot(np.squeeze(
                            error_q_numpy[:], axis=0), color='red', label='error')
                        plt.legend()
                        plt.title("Rotational error")
                        buf = io.BytesIO()
                        plt.savefig(buf, format='jpeg')
                        buf.seek(0)
                        image_q = PIL.Image.open(buf)
                        image_q = ToTensor()(image_q)
                        self.writer.add_image(
                            f"train/error_q_{index}", image_q, total_iteration_count)
                        plt.clf()

                        error_t_numpy = np.array(
                            self.errors_t[index]).T   # Epochs x 1
                        error_t_numpy = np.expand_dims(error_t_numpy, axis=0)
                        plt.plot(np.squeeze(
                            error_t_numpy[:], axis=0), color='red', label='error')
                        plt.legend()
                        plt.title("Translational error")
                        buf = io.BytesIO()
                        plt.savefig(buf, format='jpeg')
                        buf.seek(0)
                        image_t = PIL.Image.open(buf)
                        image_t = ToTensor()(image_t)
                        self.writer.add_image(
                            f"train/error_t_{index}", image_t, total_iteration_count)
                        plt.clf()
                except:
                    pass

            # Set lidar position to current estimate
            delta_tensor = torch.cat(
                (self.delta_q_list[index], self.delta_t_list[index]), axis=0)

            delta_numpy_array = delta_tensor.clone(
            ).detach().cpu().numpy()

            initial_rotation = sym.Rot3(
                initial_q[index][0, :])
            initial_pose = sym.Pose3(
                R=initial_rotation, t=initial_t[index].astype("float"))
            current_pose = sym.Pose3.retract(
                initial_pose, delta_numpy_array, _EPS)
            current_q_estimate, current_t_estimate = extract_q_t_from_pose(
                current_pose)

            R_pointcloud_camera_perturbed = quaternion_to_rotation_matrix_torch(
                current_q_estimate).cuda()
            R_pointcloud_camera_perturbed = R_pointcloud_camera_perturbed.squeeze(
                0)

            T_pointcloud_camera_perturbed = torch.vstack((torch.hstack((R_pointcloud_camera_perturbed, current_t_estimate.cuda().reshape(3, 1))),
                                                          torch.tensor([0., 0., 0., 1.]).cuda()))
            T_pointcloud_camera = T_pointcloud_camera_perturbed
            T_pointcloud_camera = T_pointcloud_camera.cuda()

            depth_map = torch.full(
                (camera_info_original.camera_height, camera_info_original.camera_width), -1.0, device="cuda")
            if lidar_pcd is not None:
                lidar_measurement = Lidar(
                    lidar_pcd.cuda(), t_lidar_camera.cuda()) 
                # DEBUG =================================================================
                # lidar_pointcloud_colmap = lidar_measurement.lidar_points_to_colmap(lidar_measurement.point_cloud)
                # ===================================================
                visible_points = lidar_measurement.lidar_points_visible(
                    lidar_measurement.point_cloud, # lidar_pointcloud_colmap,
                    T_pointcloud_camera,
                    camera_info_original.camera_intrinsics,
                    (camera_info_original.camera_width, camera_info_original.camera_height))

                depth_map = lidar_measurement.lidar_points_to_camera(
                    visible_points,
                    T_pointcloud_camera,
                    camera_info_original.camera_intrinsics,
                    (camera_info_original.camera_width,
                     camera_info_original.camera_height)
                )

            # Bundle adjustment
            depth_map_original = depth_map

            self.iterations_bundle_adjustment = 1  

            # if total_iteration_count > increase_pose_iterations_from:
            #     self.iterations_bundle_adjustment = 60

            error_q = 0.
            error_t = 0.

            for iteration in range(self.iterations_bundle_adjustment):
                image_gt = image_gt_original
                camera_info = camera_info_original
                depth_gt = depth_gt_original
                depth_map = depth_map_original

                # if total_iteration_count % self.config.half_downsample_factor_interval == 0 and total_iteration_count > 0 and downsample_factor > 1:
                if scene_iteration_count % self.config.half_downsample_factor_interval == 0 and scene_iteration_count > 0 and downsample_factor > 1:
                    downsample_factor = downsample_factor // 2
                # if downsample_factor > 1:
                if depth_map.ndim < 3:
                    depth_map = torch.unsqueeze(depth_map, axis=0)
                if downsample_factor > 1:
                    image_gt, camera_info, depth_gt, depth_map = GaussianPointTrainerBundleAdjustment._downsample_image_and_camera_info(
                        image_gt, depth_gt, camera_info, depth_map, downsample_factor=downsample_factor)

                optimizer.zero_grad()
                position_optimizer.zero_grad()
                # self.optimizers_list[index].zero_grad()
                for i in range(len(self.train_dataset)):
                    self.optimizers_q_list[i].zero_grad()
                    self.optimizers_t_list[i].zero_grad()

                delta_tensor = torch.cat(
                    (self.delta_q_list[index], self.delta_t_list[index]), axis=0).contiguous()

                if optimize_pose:
                    with torch.no_grad():
                        # Save for plotting
                        delta_numpy_array = delta_tensor.clone(
                        ).detach().cpu().numpy()
                        initial_rotation = sym.Rot3(
                            initial_q[index][0, :])
                        initial_pose = sym.Pose3(
                            R=initial_rotation, t=initial_t[index].astype("float"))
                        current_pose = sym.Pose3.retract(
                            initial_pose, delta_numpy_array, _EPS)
                        current_q, current_t = extract_q_t_from_pose(
                            current_pose)

                    q_pointcloud_camera_gt_inverse = q_pointcloud_camera_gt * \
                        np.array([-1., -1., -1., 1.])
                    q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
                        (1, 4)), current_q.reshape((1, 4)))
                    q_difference = q_difference.cpu().numpy()
                    angle_difference = np.abs(
                        2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))

                    if angle_difference > np.pi:
                        angle_difference = 2*np.pi - angle_difference
                    error_q = angle_difference
                    error_t = torch.linalg.vector_norm(
                        current_t - t_pointcloud_camera_gt)
                    self.errors_q[index].append(error_q)
                    self.errors_t[index].append(error_t)

                gaussian_point_cloud_rasterisation_input = GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    delta=delta_tensor,
                    initial_q=initial_q[index],
                    initial_t=initial_t[index],
                    color_max_sh_band=scene_iteration_count // self.config.increase_color_max_sh_band_interval,
                )

                image_pred, image_depth, pixel_valid_point_count, _ = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input)

                image_depth = image_depth.cuda()
                image_depth = image_depth/torch.max(image_depth)
                depth_gt = depth_gt / torch.max(depth_gt)

                # clip to [0, 1]
                image_pred = torch.clamp(image_pred, min=0, max=1)
                # hxwx3->3xhxw
                image_pred = image_pred.permute(2, 0, 1)

                # SAVE DEBUG  ===============================================
                if index == 10:
                    ground_truth_image_downsampled_numpy = image_gt.clone(
                    ).detach().cpu().numpy()
                    im = PIL.Image.fromarray(
                        (ground_truth_image_downsampled_numpy.transpose(1, 2, 0)*255).astype(np.uint8))
                    im.save(
                        f'/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/taichi_3d_gaussian_splatting/index10/groundtruth.png')

                    predicted_image_numpy = image_pred.clone(
                    ).detach().cpu().numpy()
                    im = PIL.Image.fromarray(
                        (predicted_image_numpy.transpose(1, 2, 0)*255).astype(np.uint8))
                    im.save(
                        f'/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/taichi_3d_gaussian_splatting/index10/prediction_{count_10}.png')
                    count_10 += 1
                # ===========================================================

                depth_mask = torch.where(depth_map >= 0, True, False)
                depth_map = depth_map / torch.max(depth_map)
                loss, l1_loss, ssim_loss, depth_loss, smooth_loss = self.loss_function(
                    image_pred,
                    image_gt,
                    image_depth,
                    depth_map,
                    depth_mask,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    pointcloud_features=self.scene.point_cloud_features)

                loss.backward()
                optimizer.step()
                position_optimizer.step()

                # # BA
                # if optimize_pose:
                #     if (not torch.isnan(self.delta_q_list[index].grad).any()):
                #         self.optimizers_q_list[index].step()
                #     if (not torch.isnan(self.delta_t_list[index].grad).any()):
                #         self.optimizers_t_list[index].step()
                #     if pose_iterations_count[index] % 10 == 0:
                #         self.q_scheduler_list[index].step()
                #         self.t_scheduler_list[index].step()

                # # Set minimum learning rate (from BAD gaussians)
                # if get_lr(self.optimizers_q_list[index]) < 1e-5:
                #     for g in self.optimizers_q_list[index].param_groups:
                #         g['lr'] = 1e-5

                # if get_lr(self.optimizers_t_list[index]) < 1e-5:
                #     for g in self.optimizers_t_list[index].param_groups:
                #         g['lr'] = 1e-5

                recent_losses.append(loss.item())

                previous_smooth = smooth_loss
                previous_depth_loss = depth_loss.item()
                previous_ssim_loss = ssim_loss.item()

                # if total_iteration_count % self.config.increase_ba_iterations_interval == 0 and self.iterations_pose < 5:
                #     if self.iterations_pose == 1:
                #         self.iterations_pose = 2
                #     else:
                #         self.iterations_pose = int(
                #             self.iterations_pose * self.iterations_pose_factor)

                if total_iteration_count > self.config.start_pose_optimization and optimize_pose is False:
                    print("Set pose optimization to true")
                    optimize_pose = True

                if scene_iteration_count % self.config.position_learning_rate_decay_interval == 0:
                    scheduler.step()
                magnitude_grad_viewspace_on_image = None
                if self.adaptive_controller.input_data is not None:
                    magnitude_grad_viewspace_on_image = self.adaptive_controller.input_data.magnitude_grad_viewspace_on_image
                    self._plot_grad_histogram(
                        self.adaptive_controller.input_data, writer=self.writer, iteration=iteration)
                    self._plot_value_histogram(
                        self.scene, writer=self.writer, iteration=iteration)
                    self.writer.add_histogram(
                        "train/pixel_valid_point_count", pixel_valid_point_count, iteration)
                self.adaptive_controller.refinement()
                if self.adaptive_controller.has_plot:
                    fig, ax = self.adaptive_controller.figure, self.adaptive_controller.ax
                    # plot image_pred in ax
                    ax.imshow(image_pred.clone().detach().cpu().numpy().transpose(
                        1, 2, 0), zorder=1, vmin=0, vmax=1)

                    self.writer.add_figure(
                        "densify_points", fig, total_iteration_count)
                    self.adaptive_controller.figure, self.adaptive_controller.ax = plt.subplots()
                    self.adaptive_controller.has_plot = False
                if total_iteration_count % self.config.log_loss_interval == 0:
                    self.writer.add_scalar(
                        "train/loss", loss.item(), total_iteration_count)

                    if optimize_pose:
                        last_q_estimate = np.array([np.array(arr)[-1]
                                                    for arr in self.errors_q if len(arr) > 0])
                        mean_last_q_estimate = np.mean(
                            last_q_estimate) if last_q_estimate.size != 0 else 0
                        self.writer.add_scalar(
                            "train/total_error_q", mean_last_q_estimate, total_iteration_count)
                        last_t_estimate = np.array([np.array(arr)[-1]
                                                    for arr in self.errors_t if len(arr) > 0])
                        mean_last_t_estimate = np.mean(
                            last_t_estimate) if last_t_estimate.size != 0 else 0
                        self.writer.add_scalar(
                            "train/total_error_t", mean_last_t_estimate, total_iteration_count)

                    self.writer.add_scalar(
                        "train/l1 loss", l1_loss.item(), total_iteration_count)
                    self.writer.add_scalar(
                        "train/ssim loss", ssim_loss.item(), total_iteration_count)
                    if (depth_loss.item() > 0):
                        self.writer.add_scalar(
                            "train/depth loss", depth_loss.item(), total_iteration_count)
                    self.writer.add_scalar(
                        "train/smooth_loss", smooth_loss, total_iteration_count)

                    if self.config.print_metrics_to_console:
                        print(f"train_iteration={total_iteration_count};")
                        print(f"train_loss={loss.item()};")
                        print(f"train_l1_loss={l1_loss.item()};")
                        print(f"train_ssim_loss={ssim_loss.item()};")

                if self.config.enable_taichi_kernel_profiler and total_iteration_count % self.config.log_taichi_kernel_profile_interval == 0 and iteration > 0:
                    ti.profiler.print_kernel_profiler_info("count")
                    ti.profiler.clear_kernel_profiler_info()
                if total_iteration_count % self.config.log_metrics_interval == 0:
                    psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                        image_pred=image_pred, image_gt=image_gt)
                    self.writer.add_scalar(
                        "train/psnr", psnr_score.item(), total_iteration_count)
                    self.writer.add_scalar(
                        "train/ssim", ssim_score.item(), total_iteration_count)
                    if self.config.print_metrics_to_console:
                        print(f"train_psnr={psnr_score.item()};")
                        print(
                            f"train_psnr_{total_iteration_count}={psnr_score.item()};")
                        print(f"train_ssim={ssim_score.item()};")
                        print(
                            f"train_ssim_{total_iteration_count}={ssim_score.item()};")

                is_problematic = False
                if len(recent_losses) == recent_losses.maxlen and total_iteration_count - previous_problematic_iteration > recent_losses.maxlen:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    if loss.item() > avg_loss * 1.5:
                        is_problematic = True
                        previous_problematic_iteration = total_iteration_count

                try:
                    if total_iteration_count % self.config.log_image_interval == 0 or is_problematic:
                        # make image_depth to be 3 channels
                        image_depth = self._easy_cmap(image_depth)
                        pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(0).repeat(3, 1, 1) / \
                            pixel_valid_point_count.max()
                        image_list = [image_pred, image_gt,
                                      image_depth, pixel_valid_point_count]
                        if magnitude_grad_viewspace_on_image is not None:
                            magnitude_grad_viewspace_on_image = magnitude_grad_viewspace_on_image.permute(
                                2, 0, 1)
                            magnitude_grad_u_viewspace_on_image = magnitude_grad_viewspace_on_image[
                                0]
                            magnitude_grad_v_viewspace_on_image = magnitude_grad_viewspace_on_image[
                                1]
                            magnitude_grad_u_viewspace_on_image /= magnitude_grad_u_viewspace_on_image.max()
                            magnitude_grad_v_viewspace_on_image /= magnitude_grad_v_viewspace_on_image.max()
                            image_diff = torch.abs(image_pred - image_gt)
                            image_list.append(
                                magnitude_grad_u_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                            image_list.append(
                                magnitude_grad_v_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                            image_list.append(image_diff)
                        grid = make_grid(image_list, nrow=2)

                        if is_problematic:
                            self.writer.add_image(
                                "image_problematic", grid, total_iteration_count)
                        else:
                            self.writer.add_image(
                                "image", grid, total_iteration_count)
                except Exception as e:
                    print("Exception")
                    print(e)

                if total_iteration_count % self.config.save_position == 0 and optimize_pose is True:
                    for i in range(len(self.errors_q)):
                        # Save position
                        entry_q = self.errors_q[i]
                        numpy_array_q = np.array(
                            [tensor.item() for tensor in entry_q])
                        np.savetxt(f'errors/error_q_{i}.txt', numpy_array_q)

                        entry_t = self.errors_t[i]
                        numpy_array_t = np.array(
                            [tensor.item() for tensor in entry_t])
                        np.savetxt(f'errors/error_t_{i}.txt', numpy_array_t)

                del image_pred, image_depth, loss, l1_loss, ssim_loss, depth_loss
                # they use 7000 in paper, it's hard to set a interval so hard code it here
                if (total_iteration_count % self.config.val_interval == 0 and total_iteration_count != 0) or total_iteration_count == 7000 or total_iteration_count == 5000:
                    self.validation(
                        val_data_loader, total_iteration_count, initial_q, initial_t)

                total_iteration_count += 1
                scene_iteration_count += 1
                pose_iterations_count[index] += 1

            print(f"TOTAL ITERATION COUNT: {total_iteration_count}")

            del image_gt, depth_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input,

        # Log sanity check results
        COLUMN_NAMES = ['index', 'groundtruth_pointcloud_camera_q', 'groundtruth_pointcloud_camera_t', 'perturbed_pointcloud_camera_q',
                        'perturbed_pointcloud_camera_t', 'delta', 'final_pointcloud_camera_q', 'final_pointcloud_camera_t']
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for i, pose in enumerate(initial_noisy_poses):
            try:
                pose = initial_noisy_poses[i]
                perturbed_pointcloud_camera_q, perturbed_pointcloud_camera_t = extract_q_t_from_pose(
                    initial_noisy_poses[i])

                delta_tensor = torch.cat(
                    (self.delta_q_list[i], self.delta_t_list[i]), axis=0)
                # delta_numpy_array = self.delta_list[i].clone(
                # ).detach().cpu().numpy()
                delta_numpy_array = delta_tensor.clone(
                ).detach().cpu().numpy()
                current_pose = sym.Pose3.retract(
                    initial_noisy_poses[i], delta_numpy_array, _EPS)
                final_pointcloud_camera_q, final_pointcloud_camera_t = extract_q_t_from_pose(
                    current_pose)
                row_data = {
                    'index': i,
                    'groundtruth_pointcloud_camera_q': groundtruth_q[i].clone().detach().cpu().numpy(),
                    'groundtruth_pointcloud_camera_t': groundtruth_t[i].clone().detach().cpu().numpy(),
                    'perturbed_pointcloud_camera_q': perturbed_pointcloud_camera_q.clone().detach().cpu().numpy(),
                    'perturbed_pointcloud_camera_t': perturbed_pointcloud_camera_t.clone().detach().cpu().numpy(),
                    'delta': self.delta_list[i].clone().detach().cpu().numpy(),
                    'final_pointcloud_camera_q': final_pointcloud_camera_q.clone().detach().cpu().numpy(),
                    'final_pointcloud_camera_t': final_pointcloud_camera_t.clone().detach().cpu().numpy()
                }
                df.loc[i] = row_data
            except Exception as e:
                row_data = {
                    'index': i,
                    'groundtruth_pointcloud_camera_q': None,
                    'groundtruth_pointcloud_camera_t': None,
                    'perturbed_pointcloud_camera_q': None,
                    'perturbed_pointcloud_camera_t': None,
                    'delta': None,
                    'final_pointcloud_camera_q': None,
                    'final_pointcloud_camera_t': None
                }
                df.loc[i] = row_data
        df.to_csv('out.csv', index=False)

        # Save optimal delta
        optimal_delta = np.squeeze(optimal_delta, axis=1)
        optimal_delta = np.array(optimal_delta)
        np.savetxt('delta.txt', optimal_delta)

    @staticmethod
    def _easy_cmap(x: torch.Tensor):
        x_rgb = torch.zeros(
            (3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
        x_rgb[0] = torch.clamp(x, 0, 0.20) / 0.20
        x_rgb[1] = torch.clamp(x - 0.2, 0, 0.5) / 0.5
        x_rgb[2] = torch.clamp(x - 0.5, 0, 0.75) / 0.75
        return 1. - x_rgb

    @staticmethod
    def _compute_pnsr_and_ssim(image_pred, image_gt):
        with torch.no_grad():
            psnr_score = 10 * \
                torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))
            ssim_score = ssim(image_pred.unsqueeze(0), image_gt.unsqueeze(
                0), data_range=1.0, size_average=True)
            return psnr_score, ssim_score

    @staticmethod
    def _plot_grad_histogram(grad_input: GaussianPointCloudPoseRasterisation.BackwardValidPointHookInput, writer, iteration):
        with torch.no_grad():
            xyz_grad = grad_input.grad_point_in_camera
            uv_grad = grad_input.grad_viewspace
            feature_grad = grad_input.grad_pointfeatures_in_camera
            q_grad = feature_grad[:, :4]
            s_grad = feature_grad[:, 4:7]
            alpha_grad = feature_grad[:, 7]
            r_grad = feature_grad[:, 8:24]
            g_grad = feature_grad[:, 24:40]
            b_grad = feature_grad[:, 40:56]
            num_overlap_tiles = grad_input.num_overlap_tiles
            num_affected_pixels = grad_input.num_affected_pixels
            writer.add_histogram("grad/xyz_grad", xyz_grad, iteration)
            writer.add_histogram("grad/uv_grad", uv_grad, iteration)
            writer.add_histogram("grad/q_grad", q_grad, iteration)
            writer.add_histogram("grad/s_grad", s_grad, iteration)
            writer.add_histogram("grad/alpha_grad", alpha_grad, iteration)
            writer.add_histogram("grad/r_grad", r_grad, iteration)
            writer.add_histogram("grad/g_grad", g_grad, iteration)
            writer.add_histogram("grad/b_grad", b_grad, iteration)
            writer.add_histogram("value/num_overlap_tiles",
                                 num_overlap_tiles, iteration)
            writer.add_histogram("value/num_affected_pixels",
                                 num_affected_pixels, iteration)

    @staticmethod
    def _plot_value_histogram(scene: GaussianPointCloudScene, writer, iteration):
        with torch.no_grad():
            valid_point_cloud = scene.point_cloud[scene.point_invalid_mask == 0]
            valid_point_cloud_features = scene.point_cloud_features[scene.point_invalid_mask == 0]
            num_valid_points = valid_point_cloud.shape[0]
            q = valid_point_cloud_features[:, :4]
            s = valid_point_cloud_features[:, 4:7]
            alpha = valid_point_cloud_features[:, 7]
            r = valid_point_cloud_features[:, 8:24]
            g = valid_point_cloud_features[:, 24:40]
            b = valid_point_cloud_features[:, 40:56]
            writer.add_scalar("value/num_valid_points",
                              num_valid_points, iteration)
            # print(f"num_valid_points={num_valid_points};")
            writer.add_histogram("value/q", q, iteration)
            writer.add_histogram("value/s", s, iteration)
            writer.add_histogram("value/alpha", alpha, iteration)
            writer.add_histogram("value/sigmoid_alpha",
                                 torch.sigmoid(alpha), iteration)
            writer.add_histogram("value/r", r, iteration)
            writer.add_histogram("value/g", g, iteration)
            writer.add_histogram("value/b", b, iteration)

    def validation(self, val_data_loader, iteration, initial_q, initial_t):
        with torch.no_grad():
            total_loss = 0.0
            total_psnr_score = 0.0
            total_ssim_score = 0.0
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            total_inference_time = 0.0
            for idx, val_data in enumerate(tqdm(val_data_loader)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, depth_gt, lidar_pcd, t_lidar_camera, T_pointcloud_camera, index, q_pointcloud_camera_gt, t_pointcloud_camera_gt = val_data
                image_gt = image_gt.cuda()
                depth_gt = depth_gt.cuda()

                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                # make taichi happy.
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)

                delta = torch.zeros((6, 1))
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    delta=delta,
                    initial_q=q_pointcloud_camera_gt.detach().cpu().numpy(),
                    initial_t=t_pointcloud_camera_gt.detach().cpu().numpy(),
                    color_max_sh_band=3
                )

                start_event.record()
                image_pred, image_depth, pixel_valid_point_count, _ = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input)

                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred = torch.clamp(image_pred, 0, 1)
                image_pred = image_pred.permute(2, 0, 1)
                pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(
                    0).repeat(3, 1, 1) / pixel_valid_point_count.max()

                image_depth = image_depth.cuda()

                image_depth = image_depth / torch.max(image_depth)
                depth_gt = depth_gt / torch.max(depth_gt)

                T_pointcloud_camera = T_pointcloud_camera.cuda()
                depth_map = torch.full(
                    (camera_info.camera_height, camera_info.camera_width), -1.0, device="cuda")

                if lidar_pcd is not None:
                    lidar_measurement = Lidar(
                        lidar_pcd.cuda(), t_lidar_camera.cuda())
                    # DEBUG =================================================================
                    # lidar_pointcloud_colmap = lidar_measurement.lidar_points_to_colmap(lidar_measurement.point_cloud)
                    # ===================================================
                    visible_points = lidar_measurement.lidar_points_visible(
                        lidar_measurement.point_cloud, #lidar_pointcloud_colmap,
                        T_pointcloud_camera,
                        camera_info.camera_intrinsics,
                        (camera_info.camera_width, camera_info.camera_height))

                    depth_map = lidar_measurement.lidar_points_to_camera(
                        visible_points,
                        T_pointcloud_camera,
                        camera_info.camera_intrinsics,
                        (camera_info.camera_width, camera_info.camera_height)
                    )

                depth_mask = torch.where(depth_map >= 0, True, False)
                depth_map = depth_map / torch.max(depth_map)
                loss, _, _, _, _ = self.loss_function(image_pred,
                                                      image_gt,
                                                      image_depth,
                                                      depth_map,
                                                      depth_mask,
                                                      )

                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                image_diff = torch.abs(image_pred - image_gt)
                total_loss += loss.item()
                total_psnr_score += psnr_score.item()
                total_ssim_score += ssim_score.item()

                image_depth_3_channels = self._easy_cmap(image_depth)
                depth_gt = torch.squeeze(depth_gt)
                depth_gt_3_channels = self._easy_cmap(depth_gt)

                depth_map = torch.squeeze(depth_map)
                depth_gt_3_channels = self._easy_cmap(depth_map)

                grid = make_grid([image_pred, image_gt, image_depth_3_channels,
                                 depth_gt_3_channels, pixel_valid_point_count, image_diff], nrow=2)
                if self.config.log_validation_image:
                    self.writer.add_image(
                        f"val/image {idx}", grid, iteration)

            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            average_inference_time = total_inference_time / \
                len(val_data_loader)

            mean_loss = total_loss / len(val_data_loader)
            mean_psnr_score = total_psnr_score / len(val_data_loader)
            mean_ssim_score = total_ssim_score / len(val_data_loader)
            self.writer.add_scalar(
                "val/loss", mean_loss, iteration)
            self.writer.add_scalar(
                "val/psnr", mean_psnr_score, iteration)
            self.writer.add_scalar(
                "val/ssim", mean_ssim_score, iteration)
            self.writer.add_scalar(
                "val/inference_time", average_inference_time, iteration)
            if self.config.print_metrics_to_console:
                print(f"val_loss={mean_loss};")
                print(f"val_psnr={mean_psnr_score};")
                print(f"val_psnr_{iteration}={mean_psnr_score};")
                print(f"val_ssim={mean_ssim_score};")
                print(f"val_ssim_{iteration}={mean_ssim_score};")
                print(f"val_inference_time={average_inference_time};")
            self.scene.to_parquet(
                os.path.join(self.config.output_model_dir, f"scene_{iteration}.parquet"))
            if mean_psnr_score > self.best_psnr_score:
                self.best_psnr_score = mean_psnr_score
                self.scene.to_parquet(
                    os.path.join(self.config.output_model_dir, f"best_scene.parquet"))
