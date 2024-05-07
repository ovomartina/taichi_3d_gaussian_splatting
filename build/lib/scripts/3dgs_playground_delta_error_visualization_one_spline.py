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

from scipy.interpolate import make_interp_spline, splev, splrep
from scipy.interpolate import BSpline


output_folder = "scripts/3dgs_playground_output"
groundtruth_trajectory_path = "scripts/data/val_path.obj"
groundtruth_poses_lie = np.load(os.path.join(
    output_folder, "groundtruth_poses_lie.out.npy"))
optimized_poses_lie = np.load(os.path.join(
    output_folder, "optimized_poses_lie.out.npy"))

f1 = pl.figure(1)
a3d = f1.add_subplot(111, projection='3d')

N = 50
spline_curves = []
continuity = 2

segments = [optimized_poses_lie[i:i+N, :]
            for i in range(0, optimized_poses_lie.shape[0], N)]

# Interpolate B-spline curves for each segment
spline_curves = []
continuity = 2  # Desired continuity in derivatives (e.g., up to second order)

for segment in segments:
    t_values = np.linspace(0, segment.shape[0]-1, 500)
    # Extract x, y, z coordinates from the segment
    x = np.array([point[3] for point in segment])
    y = np.array([point[4] for point in segment])
    z = np.array([point[5] for point in segment])
    w_0 = np.array([point[0] for point in segment])
    w_1 = np.array([point[1] for point in segment])
    w_2 = np.array([point[2] for point in segment])

    # Create B-spline curve
    out_x = splrep(np.arange(N), x )
    out_y = splrep(np.arange(N), y )
    out_z = splrep(np.arange(N), z )
    out_w_0 = splrep(np.arange(N), w_0)
    out_w_1 = splrep(np.arange(N), w_1)
    out_w_2 = splrep(np.arange(N), w_2)

    print(out_x)

    spline_curve_x = splev(t_values, out_x)
    spline_curve_y = splev(t_values, out_y)
    spline_curve_z = splev(t_values, out_z)
    spline_curve_w_0 = splev(t_values, out_w_0)
    spline_curve_w_1 = splev(t_values, out_w_1)
    spline_curve_w_2 = splev(t_values, out_w_2)

    spline_curves.append((spline_curve_x, spline_curve_y, spline_curve_z,
                         spline_curve_w_0, spline_curve_w_1, spline_curve_w_2))

a3d.scatter(optimized_poses_lie[:, 3], optimized_poses_lie[:, 4],
            optimized_poses_lie[:, 5], color="blue", s=5, label="Reconstructed path")


for spline_curve_x, spline_curve_y, spline_curve_z, spline_curve_w_0, spline_curve_w_1, spline_curve_w_2 in spline_curves:
    # Increase the number of points for smoother curves
    a3d.plot(spline_curve_x, spline_curve_y, spline_curve_z)
    for i in range(spline_curve_x.shape[0]):
        transform = sym.Pose3.from_tangent(np.array(
            [spline_curve_w_0[i], spline_curve_w_1[i], spline_curve_w_2[i], spline_curve_x[i], spline_curve_y[i], spline_curve_z[i]])).to_homogenous_matrix()
        plotCoordinateFrame.plotCoordinateFrame(a3d, transform, size=0.1, linewidth=0.1)
    
# poses_groundtruth = pp.se3(groundtruth_poses_lie)
groundtruth_poses = [sym.Pose3.from_tangent(
    groundtruth_poses_lie[i]) for i in range(groundtruth_poses_lie.shape[0])]
poses = np.array([[*groundtruth_poses[i].position(), *groundtruth_poses[i].rotation().data[:]]
                 for i in range(len(groundtruth_poses))])

poses_groundtruth = pp.SE3(list(poses))
wayposes = pp.bspline(poses_groundtruth, 0.1)
transforms = wayposes.matrix()

a3d.plot(wayposes.tensor()[:, 0], wayposes.tensor()[:, 1],
         wayposes.tensor()[:, 2], color="red", label="Groundtruth path")
a3d.scatter(groundtruth_poses_lie[:, 3], groundtruth_poses_lie[:, 4],
            groundtruth_poses_lie[:, 5], color="red", s=5, label="Groundtruth path")


a3d.legend()
a3d.set_xlabel('X')
a3d.set_ylabel('Y')
a3d.set_zlabel('Z')
a3d.set_aspect('equal')

f1.savefig(os.path.join(
    output_folder, f'figure_3d.png'))

pickle.dump(f1, open(os.path.join(
    output_folder,  f'FigureObject.pickle'), 'wb'))
