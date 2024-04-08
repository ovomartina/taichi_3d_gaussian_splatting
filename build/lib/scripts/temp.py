import os
import numpy as np
import matplotlib.pyplot as plt
import re
dirs = os.listdir("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output")
print(dirs)

errors_q = []
errors_t = []
for dir in dirs:
    if dir.startswith('epochs'):
        pattern = r'\d+'
        epoch = int(re.search(pattern, dir).group())
        print(epoch)
        #if epoch < 10 or (epoch >45 and epoch < 55):
        if epoch != 29 and epoch !=59:
            file_path = os.path.join("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output", dir)
            file_path = os.path.join(file_path, "error_q.out")
            err_q = np.loadtxt(file_path, delimiter=",").reshape(3000, 4)
            file_path = os.path.join("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output", dir)
            file_path = os.path.join(file_path, "error_t.out")
            err_t = np.loadtxt(file_path, delimiter=",").reshape(3000, 3)
            # err_t = err_t + np.array([-0.018, 0.006, 0.007]).T
            if (not np.isnan(err_q).any()) and (not np.isnan(err_t).any()):
                errors_q.append(err_q)
                errors_t.append(err_t)
        
errors_q_numpy = np.array(errors_q)
errors_t_numpy = np.array(errors_t)
# np.savetxt("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/errors_q.out", errors_q_numpy, delimiter=',')
# np.savetxt("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/errors_t.out", errors_t_numpy, delimiter=',')
errors_q_mean = np. mean(
    errors_q_numpy, axis=0)  # epochs x 4 array
errors_t_mean = np. mean(
    errors_t_numpy, axis=0)  # epochs x 3 array

print(errors_q_numpy.shape)
print(errors_q_mean.shape)
print(errors_q_mean[:, 0].shape)

plt.figure()
plt.plot(errors_q_mean[:, 0])
plt.plot(errors_q_mean[:, 1])
plt.plot(errors_q_mean[:, 2])
plt.plot(errors_q_mean[:, 3])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Rotational error")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/rot_error.png")
plt.clf()

errors_q_norm = np.linalg.norm(errors_q_mean, axis=1)
print(errors_q_norm.shape)
plt.plot(errors_q_norm)
plt.xlabel("Epoch")
plt.ylabel("Error Norm")
plt.title("Rotational error Norm")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/rot_error_norm.png")
plt.clf()

plt.plot(errors_t_mean[:, 0])
plt.plot(errors_t_mean[:, 1])
plt.plot(errors_t_mean[:, 2])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Translational error")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/trasl_error.png")
plt.clf()

errors_t_norm = np.linalg.norm(errors_t_mean, axis=1)
plt.plot(errors_t_norm)
plt.xlabel("Epoch")
plt.ylabel("Error Norm")
plt.title("Translational error Norm")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/trasl_error_norm.png")
plt.clf()

plt.plot(errors_q_mean[500:])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Rotational error")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/rot_error_hq.png")
plt.clf()

plt.plot(errors_t_mean[500:])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Translational error")
plt.savefig(
    "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/trasl_error_hq.png")
plt.clf()

