import os
import numpy as np
import matplotlib.pyplot as plt
import re

folder = "3dgs_playground_output_2000_fine"
dirs = os.listdir(f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}")
print(dirs)

errors_q = []
errors_t = []
for dir in dirs:
    if dir.startswith('epochs'):
        pattern = r'\d+'
        epoch = int(re.search(pattern, dir).group())
        print(epoch)
        #if epoch < 10 or (epoch >45 and epoch < 55):
        # if epoch != 29 and epoch !=59:
        if epoch > -1:
            file_path = os.path.join(f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}", dir)
            file_path = os.path.join(file_path, "error_q.out")
            err_q = np.loadtxt(file_path, delimiter=",").reshape(2000, 1)
            file_path = os.path.join(f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}", dir)
            file_path = os.path.join(file_path, "error_t.out")
            err_t = np.loadtxt(file_path, delimiter=",").reshape(2000, 3)
            if (not np.isnan(err_q).any()) and (not np.isnan(err_t).any()):
                errors_q.append(err_q)
                errors_t.append(err_t)
        
errors_q_numpy = np.array(errors_q)
errors_t_numpy = np.array(errors_t)

errors_q_mean = np. mean(
    errors_q_numpy, axis=0)  # epochs x 4 array
errors_t_mean = np. mean(
    errors_t_numpy, axis=0)  # epochs x 3 array

errors_t_norm = np.linalg.norm(errors_t_numpy, axis=2)
errors_t_mean_new = np.mean(
    errors_t_norm, axis=0) 
print(f"errors_t_mean_new: {errors_t_mean_new}")

print(errors_q_mean.shape)
print(errors_t_mean.shape)
np.savetxt(f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/errors_q.out", errors_q_mean, delimiter=',')
np.savetxt(f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/errors_t.out", errors_t_mean, delimiter=',')

print(errors_q_numpy.shape)
print(errors_q_mean.shape)
print(errors_q_mean[:, 0].shape)

plt.figure()
plt.plot(errors_q_mean)
# plt.plot(errors_q_mean[:, 1])
# plt.plot(errors_q_mean[:, 2])
# plt.plot(errors_q_mean[:, 3])
plt.xlabel(f"Epoch")
plt.ylabel(f"Error")
plt.title(f"Rotational error")
plt.savefig(
    f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/rot_error.png")
plt.clf()

errors_q_norm = np.linalg.norm(errors_q_mean, axis=1)
print(errors_q_norm.shape)
plt.plot(errors_q_norm)
plt.xlabel(f"Epoch")
plt.ylabel(f"Error Norm")
plt.title(f"Rotational error Norm")
plt.savefig(
    f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/rot_error_norm.png")
plt.clf()

plt.plot(errors_t_mean[:, 0])
plt.plot(errors_t_mean[:, 1])
plt.plot(errors_t_mean[:, 2])
plt.xlabel(f"Epoch")
plt.ylabel(f"Error")
plt.title(f"Translational error")
plt.savefig(
    f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/trasl_error.png")
plt.clf()

errors_t_norm = np.linalg.norm(errors_t_mean, axis=1)
print(f"errors_t_norm:{errors_t_norm}")
plt.plot(errors_t_norm)
plt.xlabel(f"Epoch")
plt.ylabel(f"Error Norm")
plt.title(f"Translational error Norm")
plt.savefig(
    f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/trasl_error_norm.png")
plt.clf()

plt.figure(figsize=(10, 6))
plt.xlabel(f"Epoch", fontsize="x-large")
plt.plot(errors_t_mean_new,label='Translation Error ')
plt.legend(loc='upper right', fontsize="x-large")
plt.ylabel('Translation [m]', fontsize="x-large")
plt.twinx()
plt.ylabel(f"Rotation [rad]", fontsize="x-large")
plt.plot(errors_q_mean, color='orange', label='Rotation Error')
plt.title(f"Translational  and Rotational Error")
plt.legend(loc=(0.7, 0.8), fontsize="x-large")
plt.grid(True)
plt.subplots_adjust(right=0.85, bottom=0.15)

plt.savefig(
    f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/{folder}/rot_trasl_error_norm.png")
plt.clf()