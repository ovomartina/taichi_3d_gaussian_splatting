import torch
import pypose as pp
import pylab as pl
import pickle
import numpy as np
import plotCoordinateFrame
import os

def load_pt_file(file_path):
    # Load the .pt file
    try:
        tensor_data = torch.load(file_path)
    except Exception as e:
        print(f"Error loading .pt file: {e}")
        return None

    # Check if the loaded data is a tensor
    if not isinstance(tensor_data, torch.Tensor):
        print("Loaded data is not a PyTorch tensor.")
        return None

    # Check if the tensor has 7 columns
    if tensor_data.dim() != 2 or tensor_data.size(1) != 7:
        print("Loaded tensor does not have the expected shape (Nx7).")
        return None

    return tensor_data


# Example usage
file_path = "scripts/continuous_trajectory_output_q_t_perturbed_z_spline/bases.pt"
output_path = "scripts/continuous_trajectory_output_q_t_perturbed_z_spline"
groundtruth_trajectory_path = "scripts/data/val_path.obj"
bases_tensor = load_pt_file(file_path)

if bases_tensor is not None:
    print("Loaded tensor shape:", bases_tensor.shape)
    print(bases_tensor)

    f1 = pl.figure(1)
    a3d = f1.add_subplot(111, projection='3d')

    f2 = pl.figure(2)
    a2d = f2.add_subplot(111)

    # Sanity check: load and plot trajectory
    with open(groundtruth_trajectory_path, 'r') as obj_file:
        lines = obj_file.readlines()
    vertices = []
    for line in lines:
        if line.startswith('v '):
            vertex = line.split()[1:]
            vertex = [float(coord) for coord in vertex]
            vertices.append(vertex)
    vertices_array = np.array(vertices)
    a3d.plot(vertices_array[:, 0], -vertices_array[:, 2],
             vertices_array[:, 1], color="green", label="Blender path")
    a2d.plot(vertices_array[:, 0], -vertices_array[:, 2],
             color="green", label="Blender path")

    for i in range(bases_tensor.shape[0]-3):
        current_bases = bases_tensor[i:i+4, :]
        
        # Plot segment estimate
        plotCoordinateFrame.plot_trajectory(
            a3d, current_bases[:, :3], color="black", linewidth=1, label="estimation", evaluate_zspline=True)
        plotCoordinateFrame.plot_trajectory_2d(
            a2d, current_bases[:, :3], color="black", linewidth=1, label="estimation", evaluate_zspline=True)
        
    a2d.scatter(bases_tensor[:,0],bases_tensor[:,1],color="gray")
    a3d.scatter(bases_tensor[:,0],bases_tensor[:,1],bases_tensor[:,2],color="gray")
    a2d.grid()
    a3d.grid()
    a2d.axis("equal")
    a3d.axis("equal")
    a2d.set_xlim([-1.5, 1])
    a2d.set_ylim([-1.5, 1])
    a3d.set_xlim([-2.5, 2])
    a3d.set_ylim([-2, 2.5])
    a3d.set_zlim([-1, 1])
    
    f1.savefig(os.path.join(output_path, f'full_path_3d.png'))
    f2.savefig(os.path.join(output_path, f'full_path_2d.png'))
