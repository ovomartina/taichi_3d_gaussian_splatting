import numpy as np
import torch
import pypose as pp
from pypose import LieTensor
from math import factorial

M = (1/6)*torch.tensor([[5, 3, -3, 1],
                        [1, 3, 3, -2],
                        [0, 0, 0, 1]], device="cpu") #cuda

M_z = torch.tensor([[2, 1, -2, 1],
                        [0, 1, 3, -2],
                        [0, 0, -1, 1]], device="cpu")/factorial(2) #cuda
_EPS = 1e-6


def evaluate_spline_bases(poses: np.array):  # poses: Nx6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(), tt)

        section = np.array([1-w[0], w[0]-w[1], w[1]-w[2], w[2]]
                           ).reshape((1, 4))  # .repeat(6, axis=0)
        A[i, :] = section
    # poses = np.reshape(poses, (num_sample_poses*6,1))

    bases, a, b, c = np.linalg.lstsq(A, poses)
    return np.array(bases).reshape(4, 6)


def evaluate_z_spline_bases(poses: np.array):  # poses: Nx6
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M_z.clone().cpu().numpy(), tt)

        section = np.array([1-w[0], w[0]-w[1], w[1]-w[2], w[2]]
                           ).reshape((1, 4))  # .repeat(6, axis=0)
        A[i, :] = section
    # poses = np.reshape(poses, (num_sample_poses*6,1))

    bases, a, b, c = np.linalg.lstsq(A, poses)
    return np.array(bases).reshape(4, 6)


# poses: Nx6
def evaluate_spline_bases_with_derivatives_constrained(poses: np.array, derivative_0: np.array, derivative_1: np.array, previous_bases: np.array):
    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    print(f"t step: {t_step}")
    if derivative_0 is not None and derivative_1 is not None:
        A = np.zeros((num_sample_poses+2, 4))
        offset = 2
    else:
        if derivative_0 is None and derivative_1 is None:
            A = np.zeros((num_sample_poses, 4))
            offset = 0
        else:
            A = np.zeros((num_sample_poses+1, 4))
            offset = 1

    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(), tt)
        if current_t == 0:
            print("t=0", w)
        section = np.array([1-w[0], w[0]-w[1], w[1]-w[2], w[2]]
                           ).reshape((1, 4))  # .repeat(6, axis=0)
        A[i+offset, :] = section
    if derivative_0 is not None and derivative_1 is not None:
        A[0, :] = np.array([-0.5, 0, 0, 0])  # coefficient derviative time t=0
        # coefficient derviative time t=1
        A[1, :] = np.array([0, -0.5, 0, 0.5])
    else:
        if derivative_0 is not None:
            A[0, :] = np.array([-0.5, 0, 0.5, 0])
        else:
            A[0, :] = np.array([0, -0.5, 0, 0.5])

    # poses = np.reshape(poses, (num_sample_poses*6,1))
    if derivative_0 is not None and derivative_1 is not None:
        poses_jacobians = np.vstack(
            [derivative_0.reshape(1, 6), derivative_1.reshape(1, 6), poses])
    else:
        if derivative_0 is not None:
            poses_jacobians = np.vstack([derivative_0.reshape(1, 6), poses])
        if derivative_1 is not None:
            poses_jacobians = np.vstack([derivative_1.reshape(1, 6), poses])
        if derivative_0 is None and derivative_1 is None:
            poses_jacobians = poses

    if previous_bases is not None:
        print(poses_jacobians.shape)
        print("Previous poses:", previous_bases.shape)
        poses_jacobians = np.vstack([poses_jacobians, previous_bases])
        A = np.vstack([A, np.hstack([np.eye(3), np.zeros((3, 1))])])  # CHECK

    print(A.shape)
    print(poses_jacobians.shape)
    bases, _, _, _ = np.linalg.lstsq(A, poses_jacobians)
    return np.array(bases).reshape(4, 6)


def evaluate_spline_bases_lsq(poses: np.array, batch_size: int, enable_zspline = False):
    segments = poses.shape[0]//batch_size
    t_step = 1/batch_size
    if segments==1:
        A = np.zeros((poses.shape[0], 4+(segments-1))) 
    else:
        A = np.zeros((poses.shape[0], 4+(segments))) #-1
    for segment_count in range(segments):
        for i in range(batch_size):
            current_t = t_step * i
            tt = np.power(current_t, np.arange(0, 4))
            if enable_zspline:
                w = np.matmul(M_z.clone().cpu().numpy(), tt)
            else:
                w = np.matmul(M.clone().cpu().numpy(), tt)

            # .repeat(6, axis=0)
            section = np.array(
                [1-w[0], w[0]-w[1], w[1]-w[2], w[2]]).reshape((1, 4))
            A[segment_count*batch_size + i, segment_count:segment_count+4] = section

    b = poses
    bases, _, _, _ = np.linalg.lstsq(A, b)
    return (bases)

# poses: Nx6, previous_bases:3x6


def evaluate_spline_bases_constrained(poses: np.array, previous_bases: np.array):

    num_sample_poses = poses.shape[0]
    t_step = 1/num_sample_poses
    A = np.zeros((num_sample_poses + 3, 4))
    for i in range(num_sample_poses):
        current_t = t_step * i
        tt = np.power(current_t, np.arange(0, 4))
        w = np.matmul(M.clone().cpu().numpy(), tt)

        section = np.array([1-w[0], w[0]-w[1], w[1]-w[2], w[2]]
                           ).reshape((1, 4))  # .repeat(6, axis=0)
        A[i, :] = section
    b = np.vstack([poses, previous_bases])
    A[-3:, :] = np.hstack([np.eye(3), np.zeros((3, 1))])
    bases, _, _, _ = np.linalg.lstsq(A, b)

    return np.array(bases).reshape(4, 6)


def moving_window_bases_evaluation(poses: np.array, batch_size: int):
    num_segments = (poses.shape[0]//batch_size)
    reconstruction_bases = np.zeros((4+(num_segments-1), 6))
    # reconstruction_bases = np.zeros((num_segments*4-3, 6))
    print("last pose: ", num_segments)
    # Sliding window to evaluate bases
    count = 0
    for i in range(0, num_segments):

        current_poses = poses[i*batch_size:(i*batch_size)+batch_size, :]
        if i == 0:
            bases = evaluate_spline_bases(current_poses)
            reconstruction_bases[i:i+4, :] = bases
        else:
            bases = evaluate_spline_bases_constrained(
                current_poses, reconstruction_bases[i:i+3, :])
            reconstruction_bases[i:i+4, :] = bases
        count += 1
        print("========================================00")
        print(reconstruction_bases)
        print("========================================00")
    return reconstruction_bases


def cubic_bspline_interpolation(
        ctrl_knots: LieTensor,
        u: torch.Tensor | torch.Tensor,
        enable_eps: bool = False,
        enable_z_spline: bool = False
) -> LieTensor:
    """Cubic B-spline interpolation with batches of four SE(3) control knots.

    Args:
        ctrl_knots: The control knots.
        u: Normalized positions on the trajectory segments. Range: [0, 1].
        enable_eps: Whether to clip the normalized position with a small epsilon to avoid possible numerical issues.

    Returns:
        The interpolated poses.
    """
    
    batch_size = ctrl_knots.shape[:-2]
    interpolations = u.shape[-1]

    # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
    # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
    if u.dim() == 1:
        u = u.tile((*batch_size, 1))  # (*batch_size, interpolations)
    if enable_eps:
        u = torch.clip(u, _EPS, 1.0 - _EPS)

    uu = u * u
    uuu = uu * u
    oos = 1.0 / 6.0  # one over six

    # t coefficients
    if enable_z_spline:
        # w = torch.matmul(M_z, torch.tensor([torch.ones_like(u),u,uu,uuu], device="cuda"))
        
        coeffs_t= torch.stack([
            -0.5*u + uu - 0.5*uuu,
            1-5*0.5*uu+1.5*uuu,
            0.5*u+2*uu-1.5*uuu,
            -0.5*uu+0.5*uuu
        ], dim=-2)

    else:
        coeffs_t = torch.stack([
            oos - 0.5 * u + 0.5 * uu - oos * uuu,
            4.0 * oos - uu + 0.5 * uuu,
            oos + 0.5 * u + 0.5 * uu - 0.5 * uuu,
            oos * uuu
        ], dim=-2)
        

    # spline t
    t_t = torch.sum(pp.bvv(coeffs_t, ctrl_knots.translation()), dim=-3)

    # q coefficients
    if enable_z_spline:
        coeffs_r = torch.stack([
            1.0 + 0.5 * u -  uu + 0.5 * uuu,
            0.5*u+1.5*uu-uuu,
            -0.5*uu+0.5*uuu
        ], dim=-2)
    else:
        coeffs_r = torch.stack([
            5.0 * oos + 0.5 * u - 0.5 * uu + oos * uuu,
            oos + 0.5 * u + 0.5 * uu - 2 * oos * uuu,
            oos * uuu
        ], dim=-2)

    # spline q
    q_adjacent = ctrl_knots[..., :-1, :].rotation().Inv() @ ctrl_knots[..., 1:, :].rotation()
    r_adjacent = q_adjacent.Log()
    q_ts = pp.Exp(pp.so3(pp.bvv(coeffs_r, r_adjacent)))
    q0 = ctrl_knots[..., 0, :].rotation()  # (*batch_size, 4)
    q_ts = torch.cat([
        q0.unsqueeze(-2).tile((interpolations, 1)).unsqueeze(-3),
        q_ts
    ], dim=-3)  # (*batch_size, num_ctrl_knots=4, interpolations, 4)
    q_t = pp.cumprod(q_ts, dim=-3, left=False)[..., -1, :, :]

    ret = pp.SE3(torch.cat([t_t, q_t], dim=-1))
    return ret

def bspline(data, u, extrapolate=False, enable_z_spline: bool = False):

    assert data.dim() >= 2, "Dimension of data should be [..., N, C]."
    # assert interval < 1.0, "The interval should be smaller than 1."
    batch = data.shape[:-2]
    Bth, N, D = data.shape[:-2], data.shape[-2], data.shape[-1]
    dargs = {'dtype': data.dtype, 'device': data.device}
    # timeline = torch.arange(0, 1, interval, **dargs)
    if u.dim() == 1:
        u = u.tile((*batch, 1))  # (*batch_size, interpolations)
    u = torch.clip(u, _EPS, 1.0 - _EPS)
    tt = u ** torch.arange(4, **dargs).view(-1, 1)
    tt=tt.to(torch.float32)
    if enable_z_spline:
        B = M_z
        B = B.cpu()
    else:
        B = torch.tensor([[5, 3,-3, 1],
                        [1, 3, 3,-2],
                        [0, 0, 0, 1]], **dargs) / 6

    dP = data[..., 0:-3, :].unsqueeze(-2)
    w = (B @ tt).unsqueeze(-1)
    index = (torch.arange(0, N-3).unsqueeze(-1) + torch.arange(0, 4)).view(-1)
    P = data[..., index, :].view(Bth + (N-3,4,1,D))
    P = (P[..., :3, :, :].Inv() * P[..., 1:, :, :]).Log()
    A = (P * w).Exp()
    Aend = (P[..., -1, :] * ((B.sum(dim=1)).unsqueeze(-1))).Exp()
    Aend = Aend[..., [0], :] * Aend[..., [1], :] * Aend[..., [2], :]
    A = A[..., 0, :, :] * A[..., 1, :, :] * A[..., 2, :, :]
    ps, pend = dP * A,dP[...,-1,:,:]*Aend[...,-1,:,:]
    poses = ps.view(Bth + (-1, D))
    return poses