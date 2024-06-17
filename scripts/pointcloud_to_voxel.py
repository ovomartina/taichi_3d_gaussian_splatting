import os
import numpy as np
import pandas as pd
import open3d as o3d
import argparse
from matplotlib import pyplot as plt
from groundtruth_centroids_evaluation import point_cloud_groundtruth_comparison, get_pointcloud_dimension
import string


def main(mesh_gt_path, mesh_reconstructed_path): # groundtruth_voxel_grid_path, reconstructed_voxel_grid_path
    
    # Convert ply to voxel
    # mesh_gt =  o3d.io.read_point_cloud("/media/scratch1/mroncoroni/groundtruth/replica/room_1/mesh_upsampled.ply")
    # mesh_reconstructed =  o3d.io.read_point_cloud("output/replica_colmap/room_1_high_quality_500_frames_bundle_adjustment_continuous_lidar_backprop/point_clouds/completePointSet_aligned_icp.ply")

    mesh_gt =  mesh_gt_path
    mesh_reconstructed =  mesh_reconstructed_path
    parent_directory = mesh_reconstructed_path.parent
    original_file_name = mesh_reconstructed_path.stem 
    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_gt,
                                                                voxel_size=0.01)
    voxel_grid_reconstruction = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_reconstructed,
                                                                voxel_size=0.01)
    print(len(voxel_grid_reconstruction.get_voxels()))
    o3d.io.write_voxel_grid("/media/scratch1/mroncoroni/groundtruth/replica/room_1/voxelized_gt.ply", voxel_grid)
    
    new_file_name = f'voxelized_{original_file_name}.ply'

    # Create the new file path
    new_file_path = os.path.join(parent_directory,new_file_name)
    o3d.io.write_voxel_grid(new_file_path, voxel_grid_reconstruction)
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--groundtruth_voxel_grid_path", type=str, required=True)
    # parser.add_argument("--reconstructed_voxel_grid_path", type=str, required=True)
    
    # args = parser.parse_args()
    # main(args.groundtruth_voxel_grid_path, args.reconstructed_voxel_grid_path)
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-gt-path', type=str, help='Parquet file path')
    parser.add_argument('--mesh-reconstructed-path', type=str, help='Json trajectory file path')
    args = parser.parse_args()
    main(args.mesh_gt_path, args.mesh_reconstructed_path)