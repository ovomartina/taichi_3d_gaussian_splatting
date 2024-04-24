import numpy as np
import torch

M = (1/6)*torch.tensor([[5, 3, -3, 1],
                        [1, 3, 3, -2],
                        [0, 0, 0, 1]], device="cuda")
def evaluate_spline_bases(poses: np.array): #poses: Nx6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(),tt)
        
        section = np.array([1-w[0], w[0]-w[1],w[1]-w[2], w[2]]).reshape((1,4)) # .repeat(6, axis=0)
        A[i,:]= section
    # poses = np.reshape(poses, (num_sample_poses*6,1))

    bases,a,b,c = np.linalg.lstsq(A, poses)
    return np.array(bases).reshape(4,6)

def evaluate_spline_bases_constrained(poses: np.array, previous_bases: np.array): #poses: Nx6, previous_bases:3x6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses + 3, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(),tt)
        
        section = np.array([1-w[0], w[0]-w[1],w[1]-w[2], w[2]]).reshape((1,4)) # .repeat(6, axis=0)
        A[i,:]= section
    b = np.vstack([poses, previous_bases])
    A[-3:,:]=np.hstack([np.eye(3), np.zeros((3,1))])
    bases,_,_,_ = np.linalg.lstsq(A, b)

    return np.array(bases).reshape(4,6)

def moving_window_bases_evaluation(poses: np.array, batch_size:int):
    num_segments = (poses.shape[0]//batch_size)
    reconstruction_bases = np.zeros((4+(num_segments-1), 6))
    # reconstruction_bases = np.zeros((num_segments*4-3, 6))
    print("last pose: ",num_segments)
    # Sliding window to evaluate bases
    count = 0
    for i in range (0, num_segments):
        
        current_poses = poses[i*batch_size:(i*batch_size)+batch_size, :]
        if i == 0:
            bases = evaluate_spline_bases(current_poses)
            reconstruction_bases[i:i+4,:] = bases
        else:
            bases = evaluate_spline_bases_constrained(current_poses, reconstruction_bases[i:i+3,:])
            reconstruction_bases[i:i+4,:] = bases
        count += 1
        print("========================================00")
        print(reconstruction_bases)
        print("========================================00")
    return reconstruction_bases