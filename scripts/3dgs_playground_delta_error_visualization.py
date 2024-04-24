# Plot estimate of continuous time trajectory
import os
import pypose as pp
import pylab as pl
import pickle
import numpy as np
import sym
from continuous_trajectory import evaluate_spline_bases
import plotCoordinateFrame
import curve_evaluation
output_folder = "scripts/3dgs_playground_output_continuous"
groundtruth_trajectory_path = "scripts/data/val_path.obj"
groundtruth_poses_lie = np.load(os.path.join(
    output_folder, "groundtruth_poses_lie.out.npy"))
optimized_poses_lie = np.load(os.path.join(
    output_folder, "optimized_poses_lie.out.npy"))

f1 = pl.figure(1)
a3d = f1.add_subplot(111, projection='3d')


N = 4
num_segments = (optimized_poses_lie.shape[0]//N)
reconstruction_bases = np.zeros((4+(num_segments-1), 6))
print("Num segments:",num_segments)
# Sliding window to evaluate bases
for i in range(0, num_segments):
    current_poses = optimized_poses_lie[i*N:(i*N)+N, :]
    print(current_poses)
    print(f"{i}:{(i+4)}")
    bases = evaluate_spline_bases(current_poses)
    if i == 0:
        reconstruction_bases[i:i+4, :] = bases
    else:
        print(f"{(i*4)-3}:{(i*4)+1}")
        reconstruction_bases[i:i+4, :] = bases
    print("Bases:", bases)

# reconstruction_bases = curve_evaluation.moving_window_bases_evaluation(optimized_poses_lie, N)

# optimized_poses = [sym.Pose3.from_tangent(optimized_poses_lie[i]) for i in range(optimized_poses_lie.shape[0])]
# poses = np.array([[ *optimized_poses[i].position(), *optimized_poses[i].rotation().data[:]] for i in range(len(optimized_poses))])

# reconstruction_bases = reconstruction_bases[0:4,:]
optimized_poses = [sym.Pose3.from_tangent(
    reconstruction_bases[i]) for i in range(reconstruction_bases.shape[0])]
poses = np.array([[*optimized_poses[i].position(), *optimized_poses[i].rotation().data[:]]
                 for i in range(len(optimized_poses))])

poses = pp.SE3(list(poses))
wayposes = pp.bspline(poses, 0.1)
transforms = wayposes.matrix()
# for t in transforms:
#     plotCoordinateFrame.plotCoordinateFrame(a3d, t)
# a3d.plot(wayposes.tensor()[:, 0], wayposes.tensor()[:, 1],
#          wayposes.tensor()[:, 2], color="blue", label="Reconstructed path")
a3d.scatter(optimized_poses_lie[:, 3], optimized_poses_lie[:, 4],
            optimized_poses_lie[:, 5], color="blue", s=5, label="Reconstructed path")

a3d.scatter(reconstruction_bases[:, 3], reconstruction_bases[:, 4],
            reconstruction_bases[:, 5], color="green", s=5, label="Reconstructed bases")

print("Reconstruction bases:", reconstruction_bases.shape[0])
print("num segments:", num_segments)
for i in range(0, reconstruction_bases.shape[0]-3):
    print(f"{i}:{i+4}")
    plotCoordinateFrame.plot_trajectory(
        a3d, reconstruction_bases[i:i+4, 3:], color="black", linewidth=1, resolution=0.1, label="estimation")
# Plot groundtruth continuous time trajectory

# poses_groundtruth = pp.se3(groundtruth_poses_lie)
groundtruth_poses = [sym.Pose3.from_tangent(
    groundtruth_poses_lie[i]) for i in range(groundtruth_poses_lie.shape[0])]
poses = np.array([[*groundtruth_poses[i].position(), *groundtruth_poses[i].rotation().data[:]]
                 for i in range(len(groundtruth_poses))])

poses_groundtruth = pp.SE3(list(poses))
wayposes = pp.bspline(poses_groundtruth, 0.1)
transforms = wayposes.matrix()
# for t in transforms:
#     plotCoordinateFrame.plotCoordinateFrame(a3d, t)

a3d.plot(wayposes.tensor()[:, 0], wayposes.tensor()[:, 1],
         wayposes.tensor()[:, 2], color="red", label="Groundtruth path")
a3d.scatter(groundtruth_poses_lie[:, 3], groundtruth_poses_lie[:, 4],
            groundtruth_poses_lie[:, 5], color="red", s=5, label="Groundtruth path")

# # Sanity check: load and plot trajectory
# with open(groundtruth_trajectory_path, 'r') as obj_file:
#     lines = obj_file.readlines()
# vertices = []
# for line in lines:
#     if line.startswith('v '):
#         vertex = line.split()[1:]
#         vertex = [float(coord) for coord in vertex]
#         vertices.append(vertex)
# vertices_array = np.array(vertices)
# a3d.plot(vertices_array[:, 0], -vertices_array[:, 2], vertices_array[:, 1], color="green", label="Blender path")

a3d.legend()
a3d.set_xlabel('X')
a3d.set_ylabel('Y')
a3d.set_zlabel('Z')
a3d.set_aspect('equal')

f1.savefig(os.path.join(
    output_folder, f'figure_3d.png'))

pickle.dump(f1, open(os.path.join(
    output_folder,  f'FigureObject.pickle'), 'wb'))
