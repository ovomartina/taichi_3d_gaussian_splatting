import os
import re
import numpy as np
import curve_evaluation
import sym
import pylab as pl
import plotCoordinateFrame
import open3d as o3d
import copy
import pypose as pp
import torch
import json
# COmpare against distorted LiDAR


def main():
    optimization_folder = "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output_50_frames_discrete_new"

    lidar_folder = "/media/scratch1/mroncoroni/groundtruth/replica/room_1/lidar_scan_high_freq"
    dirs = os.listdir(optimization_folder)
    last_epoch = 2900
    final_q = np.zeros((len(dirs), 4))
    final_t = np.zeros((len(dirs), 3))
    
    first_epoch = 0
    first_q = np.zeros((len(dirs), 4))
    first_t = np.zeros((len(dirs), 3))


    pattern = r'epochs_delta_(\d+)'
    dirs = [x for x in dirs if x.startswith('epoch')]
    file_numbers = [(filename, int(re.search(pattern, filename).group(1)))
                    for filename in dirs]

    # Sort the filenames based on the extracted numeric values
    dirs = [filename for filename, _ in sorted(
        file_numbers, key=lambda x: x[1])]
    print(dirs)

    for count, dir in enumerate(dirs):
        if count > 44:
            continue
        if dir.startswith('epochs'):
            final_q[count, :] = np.loadtxt(os.path.join(
                optimization_folder, dir, f"epoch_{last_epoch}_q.txt"))
            final_t[count, :] = np.loadtxt(os.path.join(
                optimization_folder, dir, f"epoch_{last_epoch}_t.txt"))
            
            first_q[count, :] = np.loadtxt(os.path.join(
                optimization_folder, dir, f"epoch_{first_epoch}_q.txt"))
            first_t[count, :] = np.loadtxt(os.path.join(
                optimization_folder, dir, f"epoch_{first_epoch}_t.txt"))

    # Evaluate continuous trajectory
    # Generate lie poses
    discrete_pose_lie = np.zeros((44, 6))
    first_discrete_pose_lie= np.zeros((44, 6))
    for i in range(44):
        discrete_pose_lie[i, :] = sym.Pose3.from_storage(
            np.hstack((final_q[i, :], final_t[i, :]))).to_tangent()
        
    for i in range(44):
        first_discrete_pose_lie[i, :] = sym.Pose3.from_storage(
            np.hstack((first_q[i, :], first_t[i, :]))).to_tangent()

    batch_size = 2
    
    # bases = np.vstack((discrete_pose_lie[0,:],\
    #     discrete_pose_lie[0:-1:2,:],\
    #     discrete_pose_lie[-1,:]))
    bases = discrete_pose_lie
    # bases = curve_evaluation.evaluate_spline_bases_lsq(
    #         discrete_pose_lie, batch_size)
    first_bases = first_discrete_pose_lie
    # Plot blender & GT
    # Load blender trajectory
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

    groundtruth_trajectory_path="/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/data/replica/room_1_500_frames_continuous/train.json"
    with open(groundtruth_trajectory_path) as f:
        d = json.load(f)
        vertices = []
        poses = []
        for view in d:
            groundtruth_T_pointcloud_camera = np.array(
                    view["T_pointcloud_camera"])
            rot = sym.Rot3.from_rotation_matrix(groundtruth_T_pointcloud_camera[:3,:3])
            print(np.hstack((np.array(rot.to_storage()).reshape((1,4)), groundtruth_T_pointcloud_camera[:3,3].reshape((1,3)))))
            pose = sym.Pose3.from_storage(np.hstack((np.array(rot.to_storage()).reshape((1,4)), groundtruth_T_pointcloud_camera[:3,3].reshape((1,3)))).squeeze(0))
            vertices.append(groundtruth_T_pointcloud_camera[:3,3].reshape((1,3)).squeeze(0))
            poses.append(pose)
    blender_vertices_array = np.array(vertices)

    # Plot groundtruth
    a2d.plot(blender_vertices_array[:, 0], 
             blender_vertices_array[:, 1], color="green")
    a2d.scatter(blender_vertices_array[:, 0], 
             blender_vertices_array[:, 1], color="green",s=1.5)
    # Plot discrete poses
    a2d.scatter(
        first_discrete_pose_lie[:, 3], first_discrete_pose_lie[:, 4], color="black", s=3) 

    for i in range(44):
        plotCoordinateFrame.plotCoordinateFrame(
            a2d, sym.Pose3.from_tangent(first_discrete_pose_lie[i, :]).to_homogenous_matrix(), size=0.3, linewidth=0.3)
    print(bases)
    for i in range(first_bases.shape[0]-4):
        active_bases = first_bases[i:i+4, 3:]
        plotCoordinateFrame.plot_trajectory_2d(
            a2d, active_bases, color="black", evaluate_zspline=True, linewidth=1)
    a2d.grid()
    a2d.axis("equal")
    a2d.set_xlim([-2.5, -1.5])
    a2d.set_ylim([-1.5,-0.5])
    f2.savefig(os.path.join(
        optimization_folder, f'first_trajectory_continuous_zoom.png'), dpi=300)
    a2d.clear()
    
     # Plot groundtruth
    a2d.plot(blender_vertices_array[:, 0], 
             blender_vertices_array[:, 1], color="green")
    a2d.scatter(blender_vertices_array[:, 0], 
             blender_vertices_array[:, 1], color="green",s=1.5)
    # Plot discrete poses
    a2d.scatter(
        discrete_pose_lie[:, 3], discrete_pose_lie[:, 4], color="black", s=3) 

    for i in range(44):
        plotCoordinateFrame.plotCoordinateFrame(
            a2d, sym.Pose3.from_tangent(discrete_pose_lie[i, :]).to_homogenous_matrix(), size=0.3, linewidth=0.3)
    print(bases)
    for i in range(bases.shape[0]-4):
        active_bases = bases[i:i+4, 3:]
        plotCoordinateFrame.plot_trajectory_2d(
            a2d, active_bases, color="black", evaluate_zspline=True, linewidth=1)
    a2d.grid()
    a2d.axis("equal")
    # a2d.set_xlim([0, 1])
    # a2d.set_ylim([0, 0.5])
    a2d.set_xlim([-2.5, -1.5])
    a2d.set_ylim([-1.5,-0.5])
    f2.savefig(os.path.join(
        optimization_folder, f'final_trajectory_continuous_zoom.png'), dpi=300)
    a2d.clear()
    
    
    lidar_folder = "/media/scratch1/mroncoroni/groundtruth/replica/room_1/lidar_partial_500_test_views"
    lidar_frequency = 2000 #10  # [Hz]
    lidar_batch_size = 100  # Number of scans to cover 360°
    delta_time_lidar = 1/lidar_frequency  # [s] between scans

    gyroscope_noise_density = 1.6968e-04  # [ rad / s / sqrt(Hz) ]
    accelerometer_noise_density = 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]
    gyroscope_random_walk = 1.9393e-05  # [  rad / s^2 / sqrt(Hz) ]
    accelerometer_random_walk = 3.0000e-3  # [ m / s^3 / sqrt(Hz) ]
    imu_frequency = 200  # [Hz]
    delta_time_imu = 1/imu_frequency  # [s]
    gyroscope_noise_density_discrete = gyroscope_noise_density / \
        np.sqrt(delta_time_imu)  # [ rad / s]
    accelerometer_noise_density_discrete = accelerometer_noise_density / \
        np.sqrt(delta_time_imu)  # [ m / s^2]
    gyroscope_random_walk_discrete = gyroscope_random_walk * \
        np.sqrt(delta_time_imu)  # [ rad / s]
    accelerometer_random_walk_discrete = accelerometer_random_walk * \
        np.sqrt(delta_time_imu)  # [ m / s^2]

    pointcloud_files = os.listdir(lidar_folder)
    noisy_pointcloud_files = [
        x for x in pointcloud_files if x.endswith('noise.ply')]
    pattern = r'laser_scan_frame_(\d+)_noise\.ply'
    file_numbers = [(filename, int(re.search(pattern, filename).group(1)))
                    for filename in noisy_pointcloud_files]
    # Sort the filenames based on the extracted numeric values
    noisy_pointcloud_files = [filename for filename,
                              _ in sorted(file_numbers, key=lambda x: x[1])]
    # Every second has 10 scans, 5 for each rotation
    num_batches = len(noisy_pointcloud_files)//lidar_batch_size
    
    num_batches = 1 # Only care about 1st
    # Compute chamfer sitance for each full scan
    chamfer_distances = np.zeros((num_batches, 1))
    for i in range(num_batches):
        batch_pointcloud = np.empty((0, 3))
        groundtruth_batch_pointcloud = np.empty((0, 3))
        bias_gyro = np.zeros((1, 3))
        bias_accelerometer = np.zeros((1, 3))

        for j in range(lidar_batch_size, 0, -1):

            pcd = o3d.io.read_point_cloud(os.path.join(
                lidar_folder, noisy_pointcloud_files[i*lidar_batch_size+j-1]))

            time_offset_batch = (lidar_batch_size-j)*delta_time_lidar
            w_rotation = np.random.normal(0, 1, size=(1, 3))
            w_translation = np.random.normal(0, 1, size=(1, 3))
            w_rotation_bias = np.random.normal(0, 1, size=(1, 3))
            w_translation_bias = np.random.normal(0, 1, size=(1, 3))
            error_rotation_lie = gyroscope_noise_density_discrete*w_rotation
            error_rotation_lie = (error_rotation_lie +
                                  bias_gyro)*time_offset_batch
            error_translation = accelerometer_noise_density_discrete*w_translation
            error_translation = (error_translation +
                                 bias_accelerometer)*(time_offset_batch**2)*0.5

            bias_gyro = bias_gyro + gyroscope_random_walk_discrete*w_rotation_bias
            bias_accelerometer = bias_accelerometer + \
                accelerometer_random_walk_discrete*w_translation_bias

            error_SE3 = pp.Exp(
                pp.se3(np.hstack((error_translation, error_rotation_lie))))
            T_matrix = np.reshape(pp.matrix(error_SE3).cpu().numpy(), (4, 4))
            # print(T_matrix)
            lidar_transformed = copy.deepcopy(pcd).transform(T_matrix)
            if j == 0:
                batch_pointcloud = np.asarray(lidar_transformed.points)
                groundtruth_batch_pointcloud = np.asarray(pcd.points)
            else:
                batch_pointcloud = np.vstack(
                    (batch_pointcloud, np.asarray(lidar_transformed.points)))
                groundtruth_batch_pointcloud = np.vstack(
                    (groundtruth_batch_pointcloud, np.asarray(pcd.points)))
            pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_pointcloud)
        pcd_ground_truth = o3d.geometry.PointCloud()
        pcd_ground_truth.points = o3d.utility.Vector3dVector(
            groundtruth_batch_pointcloud)
        # o3d.visualization.draw_geometries([pcd])

        dist_pc1_pc2 = pcd_ground_truth.compute_point_cloud_distance(pcd)
        dist_pc1_pc2 = np.asarray(dist_pc1_pc2)
        dist_pc2_pc1 = pcd.compute_point_cloud_distance(pcd_ground_truth)
        dist_pc2_pc1 = np.asarray(dist_pc2_pc1)
        chamfer = np.sum(dist_pc1_pc2)/len(pcd_ground_truth.points) + \
            np.sum(dist_pc2_pc1)/len(pcd.points)
        print("Chamfer distance:", chamfer)
        chamfer_distances[i] = chamfer
    mean_chamfer = chamfer_distances.mean()            
    
    print(mean_chamfer)
    # Now compute the relative pose using the bspline
    # for i, lidar_file in enumerate(noisy_pointcloud_files):

    chamfers = []
    for i in range(num_batches):
        previous_lidar_scans = []
        previous_lidar_poses = []
        gt_previous_lidar_poses = []
        for j in range(lidar_batch_size):
            offset = 70 #2 * lidar_frequency
            lidar_index = i*lidar_batch_size+j+offset #DEBUG

            if lidar_index > 100:
                continue
            
            # Select the current active bases
            active_segment =(lidar_index-offset)//10 #20
            print("active_segment: ", active_segment)
            active_bases = bases[active_segment:active_segment+4]
            current_timestamp = (lidar_index-offset -(active_segment*10))/10 # 20
            # Recover pose on spline through timestamp: t_w_lj
            active_bases_storage = np.zeros((4, 7))
            for n_base, base in enumerate(active_bases):
                pose = sym.Pose3.from_tangent(base)
                translation = pose.position()
                rotation = pose.rotation()
                active_bases_storage[n_base, :] = np.hstack(
                    (translation.reshape((1, 3)), np.array(rotation.to_storage()).reshape((1, 4))))

            current_pose_tensor = curve_evaluation.cubic_bspline_interpolation(
                pp.SE3(active_bases_storage).double(),
                u=torch.tensor([current_timestamp]).double(),
                enable_z_spline=True)
            # Use groundtruth DEBUG
            gt_current_pose_tensor = pp.SE3(np.hstack((np.array(poses[lidar_index].position()).reshape((1,3)), \
                np.array(poses[lidar_index].rotation().data[:]).reshape((1,4)))))
            print(current_pose_tensor)
            print("Lidar index: ", lidar_index)
            a2d.scatter(current_pose_tensor[0,0], current_pose_tensor[0,1], color="red",s=0.5)
            # Knowing T_w_pc, Compute T_lj_pc
            # Compute chamfer distance
            if (lidar_index+1)%lidar_batch_size==0:
                print(lidar_index+1)
                
                lidar_index= lidar_index- offset
                pcd = o3d.io.read_point_cloud(os.path.join(
                lidar_folder, noisy_pointcloud_files[lidar_index]))
                T_l5_w = np.array(current_pose_tensor.matrix().inverse()).reshape((4,4))
                
                batch_pointcloud = np.asarray(pcd.transform(T_l5_w).points)
                groundtruth_batch_pointcloud = np.asarray(pcd.transform(T_l5_w).points)
                for n_scan, scan in enumerate(previous_lidar_scans):
                    # T_world_scan = np.array(pp.SE3(previous_lidar_poses[n_scan]).matrix()).reshape((4,4))
                    # scan_w = copy.deepcopy(scan).transform(T_world_scan)
                    # scan_l5 = copy.deepcopy(scan_w).transform(T_l5_w)
                    T_w_lx = pp.SE3(previous_lidar_poses[n_scan]).matrix()
                    gt_T_lx_world =  np.array(gt_previous_lidar_poses[n_scan].matrix().inverse()).reshape((4,4))
                    T_l5_lx = T_l5_w @ np.array(T_w_lx).reshape((4,4))
                    scan_lx = copy.deepcopy(scan).transform(gt_T_lx_world)
                    scan_l5 = copy.deepcopy(scan_lx).transform(T_l5_lx)
                    
                    batch_pointcloud = np.vstack((batch_pointcloud, np.asarray(scan_l5.points)))
                    groundtruth_batch_pointcloud=np.vstack((groundtruth_batch_pointcloud,np.asarray(scan.transform(T_l5_w).points))) 
                
                groundtruth_batch_pointcloud_pcd = o3d.geometry.PointCloud()
                groundtruth_batch_pointcloud_pcd.points = o3d.utility.Vector3dVector(groundtruth_batch_pointcloud)
                groundtruth_batch_pointcloud = copy.deepcopy(groundtruth_batch_pointcloud_pcd).transform(T_l5_w)
                o3d.io.write_point_cloud(os.path.join(optimization_folder, "debug_groundtruth.pcd"), groundtruth_batch_pointcloud_pcd)
                
                batch_pointcloud_pcd = o3d.geometry.PointCloud()
                batch_pointcloud_pcd.points = o3d.utility.Vector3dVector(batch_pointcloud)
                o3d.io.write_point_cloud(os.path.join(optimization_folder, "debug.pcd"), batch_pointcloud_pcd)
                
                dist_pc1_pc2 = groundtruth_batch_pointcloud_pcd.compute_point_cloud_distance(batch_pointcloud_pcd)
                dist_pc1_pc2 = np.asarray(dist_pc1_pc2)
                dist_pc2_pc1 = batch_pointcloud_pcd.compute_point_cloud_distance(groundtruth_batch_pointcloud_pcd)
                dist_pc2_pc1 = np.asarray(dist_pc2_pc1)
                chamfer = np.sum(dist_pc1_pc2)/len(groundtruth_batch_pointcloud_pcd.points)+ np.sum(dist_pc2_pc1)/len(batch_pointcloud_pcd.points)
                print("Chamfer distance:", chamfer)

                chamfers.append(chamfer)
            else:
                pcd = o3d.io.read_point_cloud(os.path.join(
                lidar_folder, noisy_pointcloud_files[lidar_index]))
                previous_lidar_scans.append(pcd)
                previous_lidar_poses.append(current_pose_tensor)
                gt_previous_lidar_poses.append(gt_current_pose_tensor)
    print(chamfers)
    mean_chamfer = np.array(chamfers).mean()            
    
    print(mean_chamfer)
    print("IMU:",chamfer_distances.mean())
    f2.savefig(os.path.join(
        optimization_folder, f'lidar.png'), dpi=300)



if __name__ == '__main__':
    np.random.seed(42)
    main()
