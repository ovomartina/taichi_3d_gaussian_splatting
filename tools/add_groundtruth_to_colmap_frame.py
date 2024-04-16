# Script to add field to json that transforms GT blender points in colmap coordinates
import numpy as np
import pandas as pd
import argparse
import json
from taichi_3d_gaussian_splatting.utils import inverse_SE3
import torch
import os

def transform_colmap_frame(json_path: str): 
    # T_world_colmap = np.array([[-0.07269247,  0.12011554, -0.46715833, -1.61574687],
    #                             [0.48165509,  0.04348471, -
    #                                 0.06376747,  0.4119509],
    #                             [0.02594256, -0.47077614, -
    #                                 0.12508256, -0.20647023],
    #                             [0.,          0.,          0.,          1.]])
    
    T_world_colmap = np.array([[-0.07453163,  0.11889557 ,-0.4673894,  -1.59774687],
                            [ 0.48178008,  0.03977281, -0.06670892,  0.4059509 ],
                            [ 0.02184015, -0.47162058, -0.12345461, -0.21347023],
                            [ 0.,          0.,          0.,          1.        ]])
    
    scale = np.mean([np.linalg.norm(T_world_colmap[:3, 0]), np.linalg.norm(T_world_colmap[:3, 1]),
                            np.linalg.norm(T_world_colmap[:3, 2])])
        
    T_world_colmap = torch.tensor(T_world_colmap)
    T_world_colmap[:3, :3] = (1/scale) * T_world_colmap[:3, :3]
    T_world_colmap[:3, 3] = (1/scale)* T_world_colmap[:3, 3]
    T_colmap_world = inverse_SE3(T_world_colmap)
    T_colmap_world = T_colmap_world.cpu().numpy()
    
    
    with open(json_path, 'r') as json_file:
        groundtruth_data = json.load(json_file)
        
    for entry in groundtruth_data:
        print (scale)
        print(T_world_colmap)
        T_world_camera = np.array(entry["T_pointcloud_camera"])
        T_world_camera[:3, 3]= (1/scale) * T_world_camera[:3, 3]
        T_colmap_camera = np.matmul(T_colmap_world, T_world_camera)
        entry["T_colmap_camera"] = T_colmap_camera.tolist()
        print(T_colmap_camera)
    json_object = json.dumps(groundtruth_data, indent=4)
    print(json_object)
    with open(os.path.join(os.path.dirname(json_path),os.path.splitext(os.path.basename(json_path))[0])+".json", "w") as outfile:
        outfile.write(json_object)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train_path", type=str, required=True)
    args = parser.parse_args()

    transform_colmap_frame(args.json_train_path)