import argparse
import json
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudPoseRasterisation import GaussianPointCloudPoseRasterization
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch, \
    quaternion_to_rotation_matrix_torch, perturb_pose_quaternion_translation_torch
from dataclasses import dataclass
from typing import List
import torch
import numpy as np
import torch.nn.functional as F
# %%
import os
import PIL.Image
import torchvision
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
import matplotlib.pyplot as plt
import symforce.symbolic as sf

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

        self.rasteriser = GaussianPointCloudPoseRasterization(
            config=GaussianPointCloudPoseRasterization.GaussianPointCloudPoseRasterizationConfig(
                near_plane=0.001,
                far_plane=1000.,
                depth_to_sort_key_scale=100.,
                enable_depth_grad=True))

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
        count = 0
        with open(self.config.json_file_path) as f:
            d = json.load(f)
            for view in d:
                # Load groundtruth image
                ground_truth_image_path = view["image_path"]
                print(f"Loading image {ground_truth_image_path}")
                ground_truth_image_numpy = np.array(
                    PIL.Image.open(ground_truth_image_path))
                ground_truth_image = torchvision.transforms.functional.to_tensor(
                    ground_truth_image_numpy)

                ground_truth_image, resized_camera_info, _ = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image,
                                                                                                                         None,
                                                                                                                         self.camera_info,
                                                                                                                         1)
                ground_truth_image = ground_truth_image.cuda()

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
                initial_guess_q_pointcloud_camera, initial_guess_t_pointcloud_camera = perturb_pose_quaternion_translation_torch(groundtruth_q_pointcloud_camera,
                                                                                                                                 groundtruth_t_pointcloud_camera,
                                                                                                                                 0.05, 0.3)
                initial_guess_q_pointcloud_camera = torch.tensor(
                    [[0.6749, 0.5794,  -0.3524, -0.2909]], device="cuda")
                initial_guess_t_pointcloud_camera = torch.tensor(
                    [[0.2528, 0.1397,  -0.0454]], device="cuda")
                initial_guess_q_pointcloud_camera.requires_grad = True
                initial_guess_t_pointcloud_camera.requires_grad = True
                print(
                    f"Ground truth transformation world to camera, in camera frame: \n\t {groundtruth_T_pointcloud_camera}")
                print(
                    f"Initial guess q: \n\t {initial_guess_q_pointcloud_camera}")
                print(
                    f"Initial guess t: \n\t {initial_guess_t_pointcloud_camera}")

                # Save groundtruth image
                im = PIL.Image.fromarray(
                    (ground_truth_image_numpy).astype(np.uint8))
                if not os.path.exists(os.path.join(self.output_path, f'groundtruth/')):
                    os.makedirs(os.path.join(self.output_path, 'groundtruth/'))
                im.save(os.path.join(self.output_path,
                        f'groundtruth/groundtruth_{count}.png'))

                # Optimization starts
                optimizer_q = torch.optim.Adam(
                    [initial_guess_q_pointcloud_camera], lr=0.001)
                optimizer_t = torch.optim.Adam(
                    [initial_guess_t_pointcloud_camera], lr=0.001)

                num_epochs = 3000
                errors_t = []
                errors_q = []
                for epoch in range(num_epochs):
                    
                    # Add error to plot
                    initial_guess_t_pointcloud_camera_numpy = initial_guess_t_pointcloud_camera.clone().detach().cpu().numpy()
                    initial_guess_q_pointcloud_camera_numpy = initial_guess_q_pointcloud_camera.clone().detach().cpu().numpy()
                    initial_guess_q_pointcloud_camera_numpy = initial_guess_q_pointcloud_camera_numpy / np.sqrt(np.sum(initial_guess_q_pointcloud_camera_numpy**2))
                    errors_t.append(np.linalg.norm(initial_guess_t_pointcloud_camera_numpy - groundtruth_t_pointcloud_camera.cpu().numpy()))
                    errors_q.append(np.linalg.norm(initial_guess_q_pointcloud_camera_numpy - groundtruth_q_pointcloud_camera.cpu().numpy()))
                    
                    # Set the gradient to zero
                    optimizer_q.zero_grad()
                    optimizer_t.zero_grad()

                    predicted_image, _, _, _ = self.rasteriser(
                        GaussianPointCloudPoseRasterization.GaussianPointCloudPoseRasterizationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            q_pointcloud_camera=initial_guess_q_pointcloud_camera,
                            t_pointcloud_camera=initial_guess_t_pointcloud_camera,
                            camera_info=self.camera_info,
                            color_max_sh_band=3,
                        )
                    )
                    predicted_image = predicted_image.permute(2, 0, 1)
                    L1 = torch.abs(predicted_image - ground_truth_image).mean()
                    L1.backward()

                    if not torch.isnan(initial_guess_t_pointcloud_camera.grad).any():
                        torch.nn.utils.clip_grad_norm_(
                            initial_guess_t_pointcloud_camera, max_norm=1.0)
                        optimizer_t.step()
                    else:
                        print("Skipped epoch ", epoch)
                        print(previous_initial_guess_t_pointcloud_camera)
                        print(previous_initial_guess_t_pointcloud_camera)

                    if not torch.isnan(initial_guess_q_pointcloud_camera.grad).any():
                        torch.nn.utils.clip_grad_norm_(
                            initial_guess_q_pointcloud_camera, max_norm=1.0)
                        optimizer_q.step()

                    if (epoch + 1) % 50 == 0 and epoch > 100:
                        with torch.no_grad():
                            print(
                                f"============== epoch {epoch + 1} ==========================")
                            print(f"loss:{L1}")
                            q_pointcloud_camera = F.normalize(
                                initial_guess_q_pointcloud_camera, p=2, dim=-1)
                            R = quaternion_to_rotation_matrix_torch(
                                q_pointcloud_camera)
                            print("Estimated rotation")
                            print(R)
                            print(
                                f"Estimated translation: \n\t {initial_guess_t_pointcloud_camera}")
                            print(
                                f"Gradient translation: \n\t {initial_guess_t_pointcloud_camera.grad}")
                            print(
                                "Ground truth transformation world to camera, in camera frame:")
                            print(groundtruth_T_pointcloud_camera)
                            image_np = predicted_image.cpu().detach().numpy()
                            im = PIL.Image.fromarray(
                                (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                            if not os.path.exists(os.path.join(self.output_path, f'epochs/')):
                                os.makedirs(os.path.join(
                                    self.output_path, 'epochs/'))
                            im.save(os.path.join(self.output_path,
                                    f'epochs/epoch_{epoch}.png'))
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs/epoch_{epoch}_q.txt'), q_pointcloud_camera.cpu().detach().numpy())
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs/epoch_{epoch}_t.txt'), initial_guess_t_pointcloud_camera.cpu().detach().numpy())
                    previous_initial_guess_t_pointcloud_camera = initial_guess_t_pointcloud_camera.clone().detach()
                    previous_grad_t_pointcloud_camera = initial_guess_t_pointcloud_camera.grad
                plt.plot(errors_q)
                plt.plot(errors_t)
                plt.xlabel("File Index")
                plt.ylabel("Error")
                plt.title("Rotational and translational error")
                plt.savefig("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/epochs/rot_trasl_error.png")
                break  # Only optimize on the first image


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
