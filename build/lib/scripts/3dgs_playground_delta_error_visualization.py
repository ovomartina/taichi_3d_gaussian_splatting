# Plot estimate of continuous time trajectory
import os
import pypose as pp
import pylab as pl
import pickle
import numpy as np
import sym

import plotCoordinateFrame
import curve_evaluation
import continuous_trajectory

from scipy.interpolate import make_interp_spline
from scipy.interpolate import BSpline



output_folder = "scripts/3dgs_playground_output_discrete_to_continuous_working"
groundtruth_trajectory_path = "scripts/data/val_path.obj"
groundtruth_poses_lie = np.load(os.path.join(
    output_folder, "groundtruth_poses_lie.out.npy"))
optimized_poses_lie = np.load(os.path.join(
    output_folder, "optimized_poses_lie.out.npy"))

f1 = pl.figure(1)
a3d = f1.add_subplot(111, projection='3d')

N = 5

num_segments = (optimized_poses_lie.shape[0]//N)
reconstruction_bases = np.zeros((4*num_segments, 6))
# reconstruction_bases = np.zeros((4+(num_segments-1), 6))
print("optimized_poses_lie shape:", optimized_poses_lie)
print("Num segments:", num_segments)

last_pose = None
previous_bases = None
count = 0
reconstruction_bases = curve_evaluation.evaluate_spline_bases_lsq(optimized_poses_lie, N)

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

for i in range(0, num_segments):
    plotCoordinateFrame.plot_trajectory(
        a3d, reconstruction_bases[i:i+4, 3:], color="black", linewidth=1, resolution=0.001, label="estimation")
    time = np.array(range(0,10,1))
    time = 0.1*time

    for t in time:
        delta_pose = continuous_trajectory.interpolate_bspline(t, reconstruction_bases[i:i+4, :])
        plotCoordinateFrame.plotCoordinateFrame(a3d, sym.Pose3.from_tangent(delta_pose).to_homogenous_matrix(),linewidth=0.1)
    # plotCoordinateFrame.plot_trajectory_lie(a3d,reconstruction_bases[i:i+4, :],  linewidth=0.1, resolution=0.01)
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
for groundtruth_pose in groundtruth_poses_lie:
    plotCoordinateFrame.plotCoordinateFrame(a3d, sym.Pose3.from_tangent(groundtruth_pose).to_homogenous_matrix(),linewidth=1)

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
