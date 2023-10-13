#!/bin/python3

import argparse
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch
from dataclasses import dataclass
import torch
import numpy as np
from PIL import Image

class GaussianPointRenderer:
    @dataclass
    class GaussianPointRendererConfig:
        parquet_path: str
        cameras: torch.Tensor
        device: str = "cuda"
        image_height: int = 544
        image_width: int = 976
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 488.0], [0.0, 581.743, 272.0], [0.0, 0.0, 1.0]],
            device="cuda")

        def set_portrait_mode(self):
            self.image_height = 976
            self.image_width = 544
            self.camera_intrinsics = torch.tensor(
                [[1163.486, 0.0, 272.0], [0.0, 1163.486, 488.0], [0.0, 0.0, 1.0]],
                device="cuda")

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, config: GaussianPointRendererConfig) -> None:
        self.config = config
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16
        scene = GaussianPointCloudScene.from_parquet(
            config.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)
        self.cameras = self.config.cameras.to(self.config.device)
        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.8,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))

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

    def run(self, output_prefix):
        num_cameras = self.cameras.shape[0]
        for i in range(num_cameras):
            c = self.cameras[i, :, :].unsqueeze(0)
            q, t = SE3_to_quaternion_and_translation_torch(c)

            with torch.no_grad():
                image, _, _ = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        point_object_id=self.scene.point_object_id,
                        camera_info=self.camera_info,
                        q_pointcloud_camera=q,
                        t_pointcloud_camera=t,
                        color_max_sh_band=3,
                    )
                )

                img = Image.fromarray(torch.clamp(image * 255, 0, 255).byte().cpu().numpy(), 'RGB')
                img.save(f'{output_prefix}/frame_{i:03}.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--poses", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--portrait_mode", action='store_true', default=False)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)

    config = GaussianPointRenderer.GaussianPointRendererConfig(
        args.parquet_path,
        torch.load(args.poses))

    if args.portrait_mode:
        config.set_portrait_mode()

    renderer = GaussianPointRenderer(config)
    renderer.run(args.output_prefix)
