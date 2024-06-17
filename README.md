# taichi_3d_gaussian_splatting
Custom implementation of LiDAR depth loss and bundle adjustment for 3D Gaussian SPlatting.

## Run the Camera Localization code
As an intermediate step, we implemented discrete camera localization on an already trained scene.
This can be run with the following bash command:
```bash
python scripts/3dgs_discrete_localization.py  --parquet_path logs/replica_colmap/room_1_high_quality_500_frames_continuous/best_scene.parquet --json_file_path data/replica/room_1_500_frames_continuous/val.json --output_path scripts/output/3dgs_localization_output/  
```
## Run the Bundle Adjustment code
Follow the original code to set up the training image synthetic dataset, if needed.

The following command runs the bundle adjustment code on the example data from Replica:

```bash
python gaussian_point_train_bundle_adjustment.py  --train_config config/replica_room_1_high_quality_500_frames_continuous.yaml
```



## Evaluate the results
After reconstructing a 3D scene, we compare it against the voxelized groundtruth mesh for replica in order to evaluate the results.
The evaluation pipeline is as follows:

- Render depth from a collection of poses (that were not included in training) and recover the 3D point cloud from it:

```bash
python scripts/render_rgb_depth_pointcloud.py 
--parquet-path logs/replica_colmap/room_1_high_quality_500_frames_bundle_adjustment_continuous/best_scene.parquet --trajectory-path data/replica/room_1_500_frames_continuous_partial_lidar/val.json --output-path output/replica_colmap/
```
- Voxelize the recovered 3D point cloud and the ground truth mesh:
```bash
python scripts/pointcloud_to_voxel.py --mesh-gt-path <path/to/ground truth/mesh>
--mesh-reconstructed-path <output/path_to_complete_3Dpointcloud>
```
- Evaluate the number of overlapping voxels:
```bash
python scripts/voxel_comparison.py --groundtruth_voxel_grid_path <path/to/ground truth/voxel> --reconstructed_voxel_grid_path <path/to/reconstructed/voxel>  --output_directory_path <desired/folder/output>
```