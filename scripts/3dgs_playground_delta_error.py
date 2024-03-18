import argparse
import json
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudPoseRasterisation import GaussianPointCloudPoseRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch, \
    quaternion_to_rotation_matrix_torch, perturb_pose_quaternion_translation_torch
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
# %%
import os
import PIL.Image
import torchvision
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
import matplotlib.pyplot as plt

import sym


def extract_q_t_from_pose(pose3: sym.Pose3) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.tensor(
        [pose3.rotation().data[:]]).to(torch.float32)
    t = torch.tensor(
        [pose3.position()]).to(torch.float32)
    return q, t


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

                num_epochs = 3000
                errors_t = []
                errors_q = []

                rotation_groundtruth = sym.Rot3(
                    groundtruth_q_pointcloud_camera_numpy[0, :])
                pose_groundtruth = sym.Pose3(
                    R=rotation_groundtruth, t=groundtruth_t_pointcloud_camera_numpy.T.astype("float"))

                print(f"Pose groundtruth:\n\t{pose_groundtruth}")
                
                # std = 0.1, First 3 elements: rotation last 3 elements: translation
                delta_numpy_array_q = np.random.normal(0, 0.15, (3, 1))
                delta_numpy_array_t = np.random.normal(0, 0.2, (3, 1))
                delta_numpy_array = np.vstack((delta_numpy_array_q, delta_numpy_array_t))
                delta_tensor = torch.zeros(
                    (6, 1), requires_grad=True, device="cuda")

                epsilon = 0.0001
                initial_pose = sym.Pose3.retract(
                    pose_groundtruth, delta_numpy_array, epsilon)

                print(f"Initial pose:\n\t{initial_pose}")
                initial_pose_q = torch.tensor(
                    [initial_pose.rotation().data[:]]).to(torch.float32)
                initial_pose_t = torch.tensor(
                    [initial_pose.position()]).to(torch.float32)

                self.rasteriser = GaussianPointCloudPoseRasterisation(
                    config=GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationConfig(
                        near_plane=0.001,
                        far_plane=1000.,
                        depth_to_sort_key_scale=100.,
                        enable_depth_grad=True,
                        initial_pose=initial_pose))

                # Optimization starts
                optimizer_delta = torch.optim.Adam(
                    [delta_tensor], lr=0.001)
                for epoch in range(num_epochs):

                    # Add error to plot
                    with torch.no_grad():
                        epsilon = 0.0001
                        delta_numpy_array = delta_tensor.clone().detach().cpu().numpy()
                        current_pose = initial_pose.retract(
                            delta_numpy_array, epsilon=epsilon)
                        current_q, current_t = extract_q_t_from_pose(
                            current_pose)
                        current_q_numpy_array = current_q.clone().detach().cpu().numpy()
                        current_t_numpy_array = current_t.clone().detach().cpu().numpy()
                        errors_t.append(np.linalg.norm(current_t_numpy_array - groundtruth_t_pointcloud_camera.cpu().numpy()))
                        errors_q.append(np.linalg.norm(current_q_numpy_array - groundtruth_q_pointcloud_camera.cpu().numpy()))
                        

                    # Set the gradient to zero
                    optimizer_delta.zero_grad()

                    predicted_image, _, _, _ = self.rasteriser(
                        GaussianPointCloudPoseRasterisation.GaussianPointCloudPoseRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            delta=delta_tensor,
                            camera_info=self.camera_info,
                            color_max_sh_band=3,
                        )
                    )
                    predicted_image = predicted_image.permute(2, 0, 1)
                    L1 = torch.abs(predicted_image - ground_truth_image).mean()
                    L1.backward()

                    if not torch.isnan(delta_tensor.grad).any():
                        torch.nn.utils.clip_grad_norm_(
                            delta_tensor, max_norm=1.0)
                        optimizer_delta.step()

                    if epoch % 50 == 0:
                        with torch.no_grad():
                            print(
                                f"============== epoch {epoch} ==========================")
                            print(f"loss:{L1}")

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

                            im = PIL.Image.fromarray(
                                (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                            if not os.path.exists(os.path.join(self.output_path, f'epochs_delta/')):
                                os.makedirs(os.path.join(
                                    self.output_path, 'epochs_delta/'))
                            im.save(os.path.join(self.output_path,
                                    f'epochs_delta/epoch_{epoch}.png'))
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs_delta/epoch_{epoch}_q.txt'), current_q.cpu().detach().numpy())
                            np.savetxt(os.path.join(
                                self.output_path, f'epochs_delta/epoch_{epoch}_t.txt'), current_t.cpu().detach().numpy())
                plt.plot(errors_q)
                plt.plot(errors_t)
                plt.xlabel("File Index")
                plt.ylabel("Error")
                plt.title("Rotational and translational error")
                plt.savefig(
                    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/epochs_delta/rot_trasl_error.png")
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
