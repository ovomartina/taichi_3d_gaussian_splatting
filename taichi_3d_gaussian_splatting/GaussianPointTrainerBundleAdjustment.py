from .GaussianPointTrainer import cycle, GaussianPointCloudTrainer
import torch
import sym
from typing import Optional, Tuple
import numpy as np
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
from .GaussianPointCloudScene import GaussianPointCloudScene
from .GaussianPointCloudPoseRasterisation import GaussianPointCloudPoseRasterisation
from .GaussianPointAdaptiveController import GaussianPointAdaptiveController
from .LossFunction import LossFunction
from .Lidar import Lidar
from .utils import quaternion_to_rotation_matrix_torch, inverse_SE3, SE3_to_quaternion_and_translation_torch
from torchvision.utils import make_grid
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

class GaussianPointTrainerBundleAdjustment(GaussianPointCloudTrainer):
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
        start_pose_optimization: int = 7000
        save_position: int = 2000
        delta_rotation_lr: float = 1e-3
        delta_translation_lr: float = 1e-3
        save_position: int = 2000
        transform: list = field(default_factory=lambda: np.array([[-0.07269247,  0.12011554, -0.46715833, -1.61574687],
                                                                    [0.48165509,  0.04348471, -0.06376747,  0.4119509],
                                                                    [0.02594256, -0.47077614, -0.12508256, -0.20647023],
                                                                    [0.,          0.,          0.,          1.]]))

        def __post_init__(self):
            self.transform = np.array(self.transform)
        
    def __init__(self, config: TrainConfig):
        super(GaussianPointTrainerBundleAdjustment, self).__init__(config)
        for i in range(len(self.train_dataset)):
            self.delta_q_list.append(torch.zeros((3, 1), dtype=torch.float64))
            self.delta_q_list[i] = self.delta_q_list[i].cuda()
            self.delta_q_list[i].requires_grad = True

            self.delta_t_list.append(torch.zeros((3, 1), dtype=torch.float64))
            self.delta_t_list[i] = self.delta_t_list[i].cuda()
            self.delta_t_list[i].requires_grad = True

            # Separate optimization for q and t
            self.optimizers_q_list.append(torch.optim.Adam(
                [self.delta_q_list[i]], lr=self.config.delta_rotation_lr, betas=(0.9, 0.999))) 
            self.optimizers_t_list.append(torch.optim.Adam(
                [self.delta_t_list[i]], lr=self.config.delta_translation_lr, betas=(0.9, 0.999)))
            
            self.q_scheduler_list.append(torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizers_q_list[i], gamma=0.9947))
            self.t_scheduler_list.append(torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizers_t_list[i], gamma=0.9947))
        
        self.rasterisation = GaussianPointCloudPoseRasterisation(
            config=self.config.rasterisation_config,
            backward_valid_point_hook=self.adaptive_controller.update,
        )
        
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
        self.pose_estimate= [[] for _ in range(len(self.train_dataset))]
        pose_iterations_count = [0 for _ in range(len(self.train_dataset))]
        optimal_delta = [np.zeros((1, 6))
                         for _ in range(len(self.train_dataset))]
        total_iteration_count = 0
        scene_iteration_count = 0
        optimize_pose = False
        previous_smooth = None
        previous_depth_loss = None
        previous_ssim_loss = None

        while total_iteration_count < self.config.num_iterations:
            image_gt, q_pointcloud_camera_current, t_pointcloud_camera_current, camera_info, depth_gt, lidar_pcd, T_lidar_camera, T_pointcloud_camera, index, q_pointcloud_camera_gt, t_pointcloud_camera_gt = next(
                train_data_loader_iter)
            
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

                # set to GT
                initial_q[index] = noisy_q_pointcloud_camera_numpy
                initial_t[index] = noisy_t_pointcloud_camera_numpy

                # Initialize errors_q and errors_t
                q_pointcloud_camera_gt_inverse = groundtruth_q[index] * \
                    np.array([-1., -1., -1., 1.])
                q_difference = quaternion_multiply_numpy(q_pointcloud_camera_gt_inverse.reshape(
                    (1, 4)), initial_q[index].reshape((1, 4)),)
                q_difference = q_difference.cpu().numpy()
                angle_difference = np.abs(
                    2*np.arctan2(np.linalg.norm(q_difference[0, 0:3]), q_difference[0, 3]))

                if angle_difference > np.pi:
                    angle_difference = 2*np.pi - angle_difference
                error_q = angle_difference
                error_t = torch.linalg.vector_norm(
                    torch.tensor(initial_t[index]) - groundtruth_t[index])

                self.errors_q[index].append(error_q)
                self.errors_t[index].append(error_t)

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

            plt.figure()
            image_gt_original = image_gt
            camera_info_original = camera_info
            depth_gt_original = depth_gt

            pose_iterations = 10

            if optimize_pose:
                print("optimizing pose")
                
                depth_map = torch.full(
                (camera_info_original.camera_height, camera_info_original.camera_width), -1.0, device="cuda")
            
                lidar_measurement=None
                visible_points=None
                if lidar_pcd is not None:     
                    print("Lidar exists")       
                    lidar_measurement = Lidar(
                        lidar_pcd.cuda(), T_lidar_camera.cuda(), T_pointcloud_camera)
                                
                    visible_points = lidar_measurement.lidar_points_visible(
                        lidar_measurement.point_cloud, 
                        camera_info_original.camera_intrinsics,
                        (camera_info_original.camera_width, camera_info_original.camera_height))

                    depth_map = lidar_measurement.lidar_points_to_camera(
                        visible_points,
                        camera_info_original.camera_intrinsics,
                        (camera_info_original.camera_width,
                        camera_info_original.camera_height)
                    )
                    
                for _ in range(pose_iterations):

                    image_gt = image_gt_original
                    camera_info = camera_info_original
                    depth_gt = depth_gt_original

                    for i in range(len(self.train_dataset)):
                        self.optimizers_q_list[i].zero_grad()
                        self.optimizers_t_list[i].zero_grad()

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
                            print(f"Index {index}, q_pointcloud_camera_gt: \n\t{q_pointcloud_camera_gt},\n \
                                  q_pointcloud_camera_gt_inverse: \n\t {q_pointcloud_camera_gt_inverse}\n \
                                  q_difference: \n\t {q_difference},\n \
                                  current_q: {current_q},\n \
                                  angle_difference: {angle_difference}")

                        self.errors_t[index].append(error_t)
                        self.pose_estimate[index].append(current_pose)

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

                    if lidar_measurement is None:
                        loss = self.config.loss_function_config.lambda_value*torch.abs(image_pred - image_gt).mean() + (1-self.config.loss_function_config.lambda_value)*(1 - ssim(image_pred, image_gt,
                                                                                        data_range=1, size_average=True))
                    else:
                        image_depth = image_depth.cuda()
                        image_depth = image_depth / torch.max(image_depth)
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
                    
                    if (not torch.isnan(self.delta_q_list[index].grad).any()) and (not torch.isnan(self.delta_t_list[index].grad).any()):
                        self.optimizers_q_list[index].step()
                        self.optimizers_t_list[index].step()

                        delta_tensor = torch.cat(
                            (self.delta_q_list[index], self.delta_t_list[index]), axis=0).contiguous()

                    if pose_iterations_count[index] % 10 == 0:
                        self.q_scheduler_list[index].step()
                    if pose_iterations_count[index] % 10 == 0:
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

                    # Plot q and t gradients
                    try:
                        self.writer.add_histogram(
                            "grad/q_grad", self.delta_q_list[index].grad, total_iteration_count)
                        self.writer.add_histogram(
                            "grad/t_grad", self.delta_t_list[index].grad, total_iteration_count)
                    except Exception as e:
                        print("Exception plotting instagram")
                        print(e)
                        print(self.delta_q_list[index].grad)
                        print(self.delta_t_list[index].grad)

                    if total_iteration_count % self.config.log_loss_interval == 0:
                        # self.writer.add_scalar(
                        #     "train/loss", loss.item(), total_iteration_count)

                        if previous_smooth is not None and previous_depth_loss is not None and previous_ssim_loss is not None:
                            # self.writer.add_scalar(
                            #     "train/smooth_loss", previous_smooth, total_iteration_count)
                            # self.writer.add_scalar(
                            #     "train/depth loss", previous_depth_loss, total_iteration_count)
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
                        
                            current_estimate = np.array(self.pose_estimate[i])
                            np.save(os.path.join(self.config.output_model_dir,f"pose_estimate_{i}.npy"), current_estimate)

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
                # current_pose_estimate = sym.Pose3.retract(
                #     initial_noisy_poses[index], delta_numpy_array, _EPS)
                # current_q_estimate, current_t_estimate = extract_q_t_from_pose(
                #     current_pose_estimate)
                try:
                    with torch.no_grad():
                        error_q_numpy = np.array(
                            self.errors_q[index]).T   
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

            depth_map = torch.full(
                (camera_info_original.camera_height, camera_info_original.camera_width), -1.0, device="cuda")
            
            lidar_measurement=None
            visible_points=None
            if lidar_pcd is not None:     
                print("Lidar exists")       
                lidar_measurement = Lidar(
                    lidar_pcd.cuda(), T_lidar_camera.cuda(), T_pointcloud_camera)
                             
                visible_points = lidar_measurement.lidar_points_visible(
                    lidar_measurement.point_cloud, 
                    camera_info_original.camera_intrinsics,
                    (camera_info_original.camera_width, camera_info_original.camera_height))

                depth_map = lidar_measurement.lidar_points_to_camera(
                    visible_points,
                    camera_info_original.camera_intrinsics,
                    (camera_info_original.camera_width,
                     camera_info_original.camera_height)
                )

            depth_map_original = depth_map

            self.iterations_bundle_adjustment = 1

            error_q = 0.
            error_t = 0.

            for iteration in range(self.iterations_bundle_adjustment):
                image_gt = image_gt_original
                camera_info = camera_info_original
                depth_gt = depth_gt_original
                depth_map = depth_map_original

                if scene_iteration_count % self.config.half_downsample_factor_interval == 0 and scene_iteration_count > 0 and downsample_factor > 1:
                    downsample_factor = downsample_factor // 2
                if depth_map.ndim < 3:
                    depth_map = torch.unsqueeze(depth_map, axis=0)
                if downsample_factor > 1:
                    image_gt, camera_info, depth_gt, depth_map = GaussianPointCloudTrainer._downsample_image_and_camera_info(
                        image_gt, depth_gt, camera_info, depth_map, downsample_factor)

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
                image_depth = image_depth / torch.max(image_depth)
                depth_gt = depth_gt / torch.max(depth_gt)

                # clip to [0, 1]
                image_pred = torch.clamp(image_pred, min=0, max=1)
                # hxwx3->3xhxw
                image_pred = image_pred.permute(2, 0, 1)

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

                recent_losses.append(loss.item())

                previous_smooth = smooth_loss
                previous_depth_loss = depth_loss.item()
                print("DEPTH LOSS: ",depth_loss.item() )
                previous_ssim_loss = ssim_loss.item()


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

            del image_gt, depth_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input, lidar_measurement, visible_points, depth_map
    
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
                        lidar_pcd.cuda(), t_lidar_camera.cuda(), T_pointcloud_camera)

                    visible_points = lidar_measurement.lidar_points_visible(
                        lidar_measurement.point_cloud, #lidar_pointcloud_colmap,  # 
                        camera_info.camera_intrinsics,
                        (camera_info.camera_width, camera_info.camera_height))

                    depth_map = lidar_measurement.lidar_points_to_camera(
                        visible_points,
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