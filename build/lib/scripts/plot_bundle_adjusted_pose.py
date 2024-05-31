import pylab as pl
import numpy as np
import os
import plotCoordinateFrame

ba_folder = "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/logs/replica_colmap/room_1_high_quality_500_frames_bundle_adjustment_continuous"

f2 = pl.figure(2)
a2d = f2.add_subplot(111)
groundtruth_trajectory_path = "scripts/data/val_path.obj"
with open(groundtruth_trajectory_path, 'r') as obj_file:
    lines = obj_file.readlines()
vertices = []
for line in lines:
    if line.startswith('v '):
        vertex = line.split()[1:]
        vertex = [float(coord) for coord in vertex]
        vertices.append(vertex)
blender_vertices_array = np.array(vertices)

# Get a sorted list of .npy files
npy_files = sorted([file for file in os.listdir(ba_folder) if file.endswith('.npy')],
                   key=lambda x: int(x.split('_')[2].split('.')[0]))
# Load the content of each .npy file and store in a list
data_list = [np.load(os.path.join(ba_folder, file), allow_pickle=True) for file in npy_files]
min_length = min(arr.shape[0] for arr in data_list)
cropped_data_list = [arr[:min_length] for arr in data_list]
print(min_length)
output_trajectory_path = os.path.join(ba_folder,"output_trajectory")
#os.mkdir(output_trajectory_path)
for i in range(min_length):
    if i%10==0:
        a2d.plot(blender_vertices_array[:, 0], -blender_vertices_array[:, 2], color="green", label="Blender path", linewidth=0.5)
        for k, entry in enumerate(cropped_data_list):
            if k%10==0:
                a2d.scatter(entry[i].to_tangent()[3], entry[i].to_tangent()[4], color = "black", s=0.1)
                rot_matrix = np.array(entry[i].to_homogenous_matrix())
                plotCoordinateFrame.plotCoordinateFrame(a2d, rot_matrix, size = 0.3, linewidth=0.2)
        a2d.set_xlim([-4.8,1])
        a2d.set_ylim([-2.5, 1])
        f2.savefig(os.path.join(output_trajectory_path, f'trajectory_{i}.png'), dpi=600)    
        a2d.clear()
