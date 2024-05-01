import torch
import taichi as ti
from dataclasses import dataclass
from .Camera import CameraInfo
from .utils import (SE3_to_quaternion_and_translation_torch, inverse_SE3_qt_torch,
                    grad_point_probability_density_from_conic)
from .GaussianPoint3D import rotation_matrix_from_quaternion, transform_matrix_from_quaternion_and_translation
from .GaussianPointCloudRasterisation import filter_point_in_camera, generate_point_attributes_in_camera_plane, \
    generate_num_overlap_tiles, generate_point_sort_key_by_num_overlap_tiles, find_tile_start_and_end, gaussian_point_rasterisation, load_point_cloud_row_into_gaussian_point_3d
from typing import Optional, Callable
from dataclass_wizard import YAMLWizard
import sym
import symforce.symbolic as sf
import numpy as np

from typing import Tuple

mat4x0f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat3x0f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat1x3f = ti.types.matrix(n=1, m=3, dtype=ti.f32)
mat1x4f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat2x3f = ti.types.matrix(n=2, m=3, dtype=ti.f32)
mat2x4f = ti.types.matrix(n=2, m=4, dtype=ti.f32)
mat3x1f = ti.types.matrix(n=3, m=1, dtype=ti.f32)
mat3x3f = ti.types.matrix(n=3, m=3, dtype=ti.f32)
mat3x4f = ti.types.matrix(n=3, m=4, dtype=ti.f32)

_EPS = 1e-6

M = (1/6)*torch.tensor([[5, 3, -3, 1],
                        [1, 3, 3, -2],
                        [0, 0, 0, 1]], device="cuda")


def extract_q_t_from_pose(pose3: sym.Pose3) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.tensor(
        [pose3.rotation().data[:]]).to(torch.float32)
    t = torch.tensor(
        [pose3.position()]).to(torch.float32)
    return q, t


@ti.kernel
def gaussian_point_rasterisation_backward_with_pose(
    camera_height: ti.i32,
    camera_width: ti.i32,
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    point_object_id: ti.types.ndarray(ti.i32, ndim=1),  # (N)
    q_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 4)
    t_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
    q_pointcloud_camera: ti.types.ndarray(ti.f32, ndim=2),  # (1, 4)
    t_pointcloud_camera: ti.types.ndarray(ti.f32, ndim=2),  # (1, 3)
    grad_q: ti.types.ndarray(ti.f32, ndim=1),  # (4)
    grad_t: ti.types.ndarray(ti.f32, ndim=1),  # (3)
    grad_q_depth: ti.types.ndarray(ti.f32, ndim=1),  # (4)
    grad_t_depth: ti.types.ndarray(ti.f32, ndim=1),  # (3)
    # (tiles_per_row * tiles_per_col)
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),  # (K)
    point_id_in_camera_list: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    rasterized_image_grad: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    enable_depth_grad: ti.template(),
    rasterized_depth_grad: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    accumulated_alpha_grad: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    rasterized_depth: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    grad_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    grad_pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    grad_uv: ti.types.ndarray(ti.f32, ndim=2),  # (N, 2)

    in_camera_grad_uv_cov_buffer: ti.types.ndarray(ti.f32, ndim=2),
    in_camera_grad_color_buffer: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    in_camera_grad_depth_buffer: ti.types.ndarray(ti.f32, ndim=1),  # (M)

    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_conic: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    point_color: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)

    need_extra_info: ti.template(),
    magnitude_grad_viewspace: ti.types.ndarray(ti.f32, ndim=1),  # (N)
    # (H, W, 2)
    magnitude_grad_viewspace_on_image: ti.types.ndarray(ti.f32, ndim=3),
    # (M, 2, 2)
    in_camera_num_affected_pixels: ti.types.ndarray(ti.i32, ndim=1),  # (M)
):
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])

    ti.loop_config(block_dim=256)
    for pixel_offset in ti.ndrange(camera_height * camera_width):
        # each block handles one tile, so tile_id is actually block_id
        tile_id = pixel_offset // 256
        thread_id = pixel_offset % 256
        tile_u = ti.cast(tile_id % (camera_width // 16), ti.i32)
        tile_v = ti.cast(tile_id // (camera_width // 16), ti.i32)

        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        tile_point_count = end_offset - start_offset

        tile_point_uv = ti.simt.block.SharedArray(
            (2, 256), dtype=ti.f32)  # 2KB shared memory
        tile_point_uv_conic = ti.simt.block.SharedArray(
            (3, 256), dtype=ti.f32)  # 4KB shared memory
        tile_point_color = ti.simt.block.SharedArray(
            (3, 256), dtype=ti.f32)  # 3KB shared memory
        tile_point_alpha = ti.simt.block.SharedArray(
            (256,), dtype=ti.f32)  # 1KB shared memory
        tile_point_depth = ti.simt.block.SharedArray(
            (ti.static(256 if enable_depth_grad else 0),), dtype=ti.f32)  # 1KB shared memory

        pixel_offset_in_tile = pixel_offset - tile_id * 256
        pixel_offset_u_in_tile = pixel_offset_in_tile % 16
        pixel_offset_v_in_tile = pixel_offset_in_tile // 16
        pixel_u = tile_u * 16 + pixel_offset_u_in_tile
        pixel_v = tile_v * 16 + pixel_offset_v_in_tile
        last_effective_point = pixel_offset_of_last_effective_point[pixel_v, pixel_u]
        org_accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        accumulated_alpha_grad_value: ti.f32 = accumulated_alpha_grad[pixel_v, pixel_u]
        d_pixel: ti.f32 = rasterized_depth[pixel_v, pixel_u]
        T_i = 1.0 - accumulated_alpha  # T_i = \prod_{j=1}^{i-1} (1 - a_j)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} \sum_{j=i+1}^{n} c_j a_j T(j)
        # let w_i = \sum_{j=i+1}^{n} c_j a_j T(j)
        # we have w_n = 0, w_{i-1} = w_i + c_i a_i T(i)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
        w_i = ti.math.vec3(0.0, 0.0, 0.0)
        depth_w_i = 0.0
        acc_alpha_w_i = 0.0

        pixel_rgb_grad = ti.math.vec3(
            rasterized_image_grad[pixel_v, pixel_u, 0], rasterized_image_grad[pixel_v, pixel_u, 1], rasterized_image_grad[pixel_v, pixel_u, 2])
        pixel_depth_grad = rasterized_depth_grad[pixel_v,
                                                 pixel_u] if enable_depth_grad else 0.0
        total_magnitude_grad_viewspace_on_image = ti.math.vec2(0.0, 0.0)

        # for inverse_point_offset in range(effective_point_count):
        # taichi only supports range() with start and end
        # for inverse_point_offset_base in range(0, tile_point_count, 256):
        num_point_blocks = (tile_point_count + 255) // 256
        for point_block_id in range(num_point_blocks):
            inverse_point_offset_base = point_block_id * 256
            block_end_idx_point_offset_with_sort_key = end_offset - inverse_point_offset_base
            block_start_idx_point_offset_with_sort_key = ti.max(
                block_end_idx_point_offset_with_sort_key - 256, 0)
            # in the later loop, we will handle the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key)
            # so we need to load the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key - 1]
            to_load_idx_point_offset_with_sort_key = block_end_idx_point_offset_with_sort_key - thread_id - 1
            if to_load_idx_point_offset_with_sort_key >= block_start_idx_point_offset_with_sort_key:
                to_load_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                to_load_uv = ti.math.vec2(
                    [point_uv[to_load_point_offset, 0], point_uv[to_load_point_offset, 1]])

                if enable_depth_grad:
                    tile_point_depth[thread_id] = point_in_camera[to_load_point_offset, 2]

                for i in ti.static(range(2)):
                    tile_point_uv[i, thread_id] = to_load_uv[i]

                for i in ti.static(range(3)):
                    tile_point_uv_conic[i,
                                        thread_id] = point_uv_conic[to_load_point_offset, i]
                for i in ti.static(range(3)):
                    tile_point_color[i,
                                     thread_id] = point_color[to_load_point_offset, i]

                tile_point_alpha[thread_id] = point_alpha_after_activation[to_load_point_offset]

            ti.simt.block.sync()
            max_inverse_point_offset_offset = ti.min(
                256, tile_point_count - inverse_point_offset_base)
            for inverse_point_offset_offset in range(max_inverse_point_offset_offset):
                inverse_point_offset = inverse_point_offset_base + inverse_point_offset_offset

                idx_point_offset_with_sort_key = end_offset - inverse_point_offset - 1
                if idx_point_offset_with_sort_key >= last_effective_point:
                    continue

                idx_point_offset_with_sort_key_in_block = inverse_point_offset_offset
                uv = ti.math.vec2(tile_point_uv[0, idx_point_offset_with_sort_key_in_block],
                                  tile_point_uv[1, idx_point_offset_with_sort_key_in_block])
                uv_conic = ti.math.vec3([
                    tile_point_uv_conic[0,
                                        idx_point_offset_with_sort_key_in_block],
                    tile_point_uv_conic[1,
                                        idx_point_offset_with_sort_key_in_block],
                    tile_point_uv_conic[2,
                                        idx_point_offset_with_sort_key_in_block],
                ])

                point_alpha_after_activation_value = tile_point_alpha[
                    idx_point_offset_with_sort_key_in_block]

                # d_p_d_mean is (2,), d_p_d_cov is (2, 2), needs to be flattened to (4,)
                gaussian_alpha, d_p_d_mean, d_p_d_cov = grad_point_probability_density_from_conic(
                    xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                    gaussian_mean=uv,
                    conic=uv_conic,
                )
                prod_alpha = gaussian_alpha * point_alpha_after_activation_value
                # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
                # 255 ) and also clamp 𝛼 with 0.99 from above.
                if prod_alpha >= 1. / 255.:
                    alpha: ti.f32 = ti.min(prod_alpha, 0.99)
                    color = ti.math.vec3([
                        tile_point_color[0,
                                         idx_point_offset_with_sort_key_in_block],
                        tile_point_color[1,
                                         idx_point_offset_with_sort_key_in_block],
                        tile_point_color[2, idx_point_offset_with_sort_key_in_block]])

                    # accumulated_alpha_i = 1. - T_i #alpha after passing current point
                    # Transmittance before passing current point
                    T_i = T_i / (1. - alpha)
                    accumulated_alpha = 1. - T_i  # accumulated alha before passing current point

                    # print(
                    #     f"({pixel_v}, {pixel_u}, {point_offset}, {point_offset - start_offset}), accumulated_alpha: {accumulated_alpha}")

                    d_pixel_rgb_d_color = alpha * T_i
                    point_grad_color = d_pixel_rgb_d_color * pixel_rgb_grad

                    # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
                    alpha_grad_from_rgb = (color * T_i - w_i / (1. - alpha)) \
                        * pixel_rgb_grad
                    # w_{i-1} = w_i + c_i a_i T(i)
                    w_i += color * alpha * T_i
                    alpha_grad: ti.f32 = alpha_grad_from_rgb.sum()

                    if enable_depth_grad:
                        depth_i = tile_point_depth[idx_point_offset_with_sort_key_in_block]
                        alpha_grad_from_depth = (depth_i * T_i - depth_w_i / (1. - alpha)) \
                            * pixel_depth_grad
                        depth_w_i += depth_i * alpha * T_i
                        alpha_grad += alpha_grad_from_depth

                    point_alpha_after_activation_grad = alpha_grad * gaussian_alpha
                    gaussian_point_3d_alpha_grad = point_alpha_after_activation_grad * \
                        (1. - point_alpha_after_activation_value) * \
                        point_alpha_after_activation_value
                    gaussian_alpha_grad = alpha_grad * point_alpha_after_activation_value
                    # gaussian_alpha_grad is dp
                    point_viewspace_grad = gaussian_alpha_grad * \
                        d_p_d_mean  # (2,) as the paper said, view space gradient is used for detect candidates for densification
                    total_magnitude_grad_viewspace_on_image += ti.abs(
                        point_viewspace_grad)
                    point_uv_cov_grad = gaussian_alpha_grad * \
                        d_p_d_cov  # (2, 2)

                    point_offset = point_offset_with_sort_key[idx_point_offset_with_sort_key]
                    point_id = point_id_in_camera_list[point_offset]
                    # atomic accumulate on block shared memory shall be faster
                    for i in ti.static(range(2)):
                        ti.atomic_add(
                            grad_uv[point_id, i], point_viewspace_grad[i])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 0],
                                  point_uv_cov_grad[0, 0])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 1],
                                  point_uv_cov_grad[0, 1])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 2],
                                  point_uv_cov_grad[1, 1])
                    if enable_depth_grad:
                        point_depth_grad = alpha * T_i * pixel_depth_grad
                        ti.atomic_add(
                            in_camera_grad_depth_buffer[point_offset], point_depth_grad)

                    for i in ti.static(range(3)):
                        ti.atomic_add(
                            in_camera_grad_color_buffer[point_offset, i], point_grad_color[i])
                    ti.atomic_add(
                        grad_pointcloud_features[point_id, 7], gaussian_point_3d_alpha_grad)

                    if need_extra_info:
                        magnitude_point_grad_viewspace = ti.sqrt(
                            point_viewspace_grad[0] ** 2 + point_viewspace_grad[1] ** 2)
                        ti.atomic_add(
                            magnitude_grad_viewspace[point_id], magnitude_point_grad_viewspace)
                        ti.atomic_add(
                            in_camera_num_affected_pixels[point_offset], 1)
            # end of the 256 block loop
            ti.simt.block.sync()
        # end of the backward traversal loop, from last point to first point
        if need_extra_info:
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              0] = total_magnitude_grad_viewspace_on_image[0]
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              1] = total_magnitude_grad_viewspace_on_image[1]
    # end of per pixel loop

    q_pointcloud_camera_taichi = ti.math.vec4(
        q_pointcloud_camera[0, 0], q_pointcloud_camera[0, 1], q_pointcloud_camera[0, 2], q_pointcloud_camera[0, 3],)
    # q_pointcloud_camera_taichi = ti.Vector(
    #                 [q_pointcloud_camera[:, idx] for idx in ti.static(range(4))])
    # t_pointcloud_camera_taichi = ti.Vector(
    #                     [t_pointcloud_camera[:, idx] for idx in ti.static(range(3))])
    # one more loop to compute the gradient from viewspace to 3D point
    for idx in range(point_id_in_camera_list.shape[0]):
        point_id = point_id_in_camera_list[idx]
        gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        point_grad_uv = ti.math.vec2(
            grad_uv[point_id, 0], grad_uv[point_id, 1])
        point_grad_uv_cov_flat = ti.math.vec4(
            in_camera_grad_uv_cov_buffer[idx, 0],
            in_camera_grad_uv_cov_buffer[idx, 1],
            in_camera_grad_uv_cov_buffer[idx, 1],
            in_camera_grad_uv_cov_buffer[idx, 2],
        )
        point_grad_depth = in_camera_grad_depth_buffer[idx] if enable_depth_grad else 0.

        point_grad_color = ti.math.vec3(
            in_camera_grad_color_buffer[idx, 0],
            in_camera_grad_color_buffer[idx, 1],
            in_camera_grad_color_buffer[idx, 2],
        )
        point_q_camera_pointcloud = ti.Vector(
            [q_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(4))])
        point_t_camera_pointcloud = ti.Vector(
            [t_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        ray_origin = ti.Vector(
            [t_pointcloud_camera[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        T_camera_pointcloud_mat = transform_matrix_from_quaternion_and_translation(
            q=point_q_camera_pointcloud,
            t=point_t_camera_pointcloud,
        )
        translation_camera = ti.Vector([
            point_in_camera[idx, j] for j in ti.static(range(3))])
        d_uv_d_translation = gaussian_point_3d.project_to_camera_position_jacobian(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )  # (2, 3)

        # -------------------------------------------------------------
        # Pose optimization code
        w_t_w_point = gaussian_point_3d.translation
        R_guess = rotation_matrix_from_quaternion(q_pointcloud_camera_taichi)

        # d_uv_d_translation_camera = d_uv_d_translation
        d_uv_d_translation_camera = project_to_camera_relative_position_jacobian(
            w_t_w_point, T_camera_pointcloud_mat, camera_intrinsics_mat)
        dR_dqx, dR_dqy, dR_dqz, dR_dqw = quaternion_to_rotation_matrix_torch_jacobian(
            (q_pointcloud_camera[0, 0], q_pointcloud_camera[0, 1], q_pointcloud_camera[0, 2], q_pointcloud_camera[0, 3]))

        # x, y, z: coordinate of points in camera frame
        dx_dq = mat1x4f([[w_t_w_point[0] * dR_dqx[0, 0] + w_t_w_point[1] * dR_dqx[1, 0] + w_t_w_point[2] * dR_dqx[2, 0] - t_pointcloud_camera[0, 0] * dR_dqx[0, 0] - t_pointcloud_camera[0, 1] * dR_dqx[1, 0] - t_pointcloud_camera[0, 2] * dR_dqx[2, 0]],
                         [w_t_w_point[0] * dR_dqy[0, 0] + w_t_w_point[1] * dR_dqy[1, 0] + w_t_w_point[2] * dR_dqy[2, 0] - t_pointcloud_camera[0, 0]
                             * dR_dqy[0, 0] - t_pointcloud_camera[0, 1] * dR_dqy[1, 0] - t_pointcloud_camera[0, 2] * dR_dqy[2, 0]],
                         [w_t_w_point[0] * dR_dqz[0, 0] + w_t_w_point[1] * dR_dqz[1, 0] + w_t_w_point[2] * dR_dqz[2, 0] - t_pointcloud_camera[0, 0]
                             * dR_dqz[0, 0] - t_pointcloud_camera[0, 1] * dR_dqz[1, 0] - t_pointcloud_camera[0, 2] * dR_dqz[2, 0]],
                         [w_t_w_point[0] * dR_dqw[0, 0] + w_t_w_point[1] * dR_dqw[1, 0] + w_t_w_point[2] * dR_dqw[2, 0] - t_pointcloud_camera[0, 0] * dR_dqw[0, 0] - t_pointcloud_camera[0, 1] * dR_dqw[1, 0] - t_pointcloud_camera[0, 2] * dR_dqw[2, 0]]])
        dy_dq = mat1x4f([[w_t_w_point[0] * dR_dqx[0, 1] + w_t_w_point[1] * dR_dqx[1, 1] + w_t_w_point[2] * dR_dqx[2, 1] - t_pointcloud_camera[0, 0] * dR_dqx[0, 1] - t_pointcloud_camera[0, 1] * dR_dqx[1, 1] - t_pointcloud_camera[0, 2] * dR_dqx[2, 1]],
                         [w_t_w_point[0] * dR_dqy[0, 1] + w_t_w_point[1] * dR_dqy[1, 1] + w_t_w_point[2] * dR_dqy[2, 1] - t_pointcloud_camera[0, 0]
                             * dR_dqy[0, 1] - t_pointcloud_camera[0, 1] * dR_dqy[1, 1] - t_pointcloud_camera[0, 2] * dR_dqy[2, 1]],
                         [w_t_w_point[0] * dR_dqz[0, 1] + w_t_w_point[1] * dR_dqz[1, 1] + w_t_w_point[2] * dR_dqz[2, 1] - t_pointcloud_camera[0, 0]
                             * dR_dqz[0, 1] - t_pointcloud_camera[0, 1] * dR_dqz[1, 1] - t_pointcloud_camera[0, 2] * dR_dqz[2, 1]],
                         [w_t_w_point[0] * dR_dqw[0, 1] + w_t_w_point[1] * dR_dqw[1, 1] + w_t_w_point[2] * dR_dqw[2, 1] - t_pointcloud_camera[0, 0] * dR_dqw[0, 1] - t_pointcloud_camera[0, 1] * dR_dqw[1, 1] - t_pointcloud_camera[0, 2] * dR_dqw[2, 1]]])
        dz_dq = mat1x4f([[w_t_w_point[0] * dR_dqx[0, 2] + w_t_w_point[1] * dR_dqx[1, 2] + w_t_w_point[2] * dR_dqx[2, 2] - t_pointcloud_camera[0, 0] * dR_dqx[0, 2] - t_pointcloud_camera[0, 1] * dR_dqx[1, 2] - t_pointcloud_camera[0, 2] * dR_dqx[2, 2]],
                         [w_t_w_point[0] * dR_dqy[0, 2] + w_t_w_point[1] * dR_dqy[1, 2] + w_t_w_point[2] * dR_dqy[2, 2] - t_pointcloud_camera[0, 0]
                             * dR_dqy[0, 2] - t_pointcloud_camera[0, 1] * dR_dqy[1, 2] - t_pointcloud_camera[0, 2] * dR_dqy[2, 2]],
                         [w_t_w_point[0] * dR_dqz[0, 2] + w_t_w_point[1] * dR_dqz[1, 2] + w_t_w_point[2] * dR_dqz[2, 2] - t_pointcloud_camera[0, 0]
                             * dR_dqz[0, 2] - t_pointcloud_camera[0, 1] * dR_dqz[1, 2] - t_pointcloud_camera[0, 2] * dR_dqz[2, 2]],
                         [w_t_w_point[0] * dR_dqw[0, 2] + w_t_w_point[1] * dR_dqw[1, 2] + w_t_w_point[2] * dR_dqw[2, 2] - t_pointcloud_camera[0, 0] * dR_dqw[0, 2] - t_pointcloud_camera[0, 1] * dR_dqw[1, 2] - t_pointcloud_camera[0, 2] * dR_dqw[2, 2]]])

        d_translation_camera_d_q = mat3x4f([[dx_dq[0, 0], dx_dq[0, 1], dx_dq[0, 2], dx_dq[0, 3]],
                                            [dy_dq[0, 0], dy_dq[0, 1],
                                                dy_dq[0, 2], dy_dq[0, 3]],
                                            [dz_dq[0, 0], dz_dq[0, 1], dz_dq[0, 2], dz_dq[0, 3]]])

        dxyz_d_t_world_camera = -R_guess.transpose()
        point_grad_q = d_uv_d_translation_camera @ d_translation_camera_d_q  # d_uv_d_q
        point_grad_t = d_uv_d_translation_camera @ dxyz_d_t_world_camera  # d_uv_d_t

        # Backpropagation on color loss
        multiply = point_grad_uv @ point_grad_q
        multiply_t = point_grad_uv @ point_grad_t

        # Backpropagation on depth loss
        # point_grad_depth: in-camera depth gradient
        # pixel_depth_grad
        # d_depth_d_translation = gaussian_point_3d.depth_jacobian(
        #         T_camera_world=T_camera_pointcloud_mat,
        #     )
        d_depth_d_translation_camera = mat1x3f([0.,
                                                0.,
                                                1.])
        # Added extra d_depth_d_translation
        d_depth_d_q = point_grad_depth * \
            d_depth_d_translation_camera @ d_translation_camera_d_q
        d_depth_d_t = point_grad_depth * \
            d_depth_d_translation_camera @ dxyz_d_t_world_camera

        grad_q[0] = grad_q[0] + multiply[0]
        grad_q[1] = grad_q[1] + multiply[1]
        grad_q[2] = grad_q[2] + multiply[2]
        grad_q[3] = grad_q[3] + multiply[3]

        grad_t[0] = grad_t[0] + multiply_t[0]
        grad_t[1] = grad_t[1] + multiply_t[1]
        grad_t[2] = grad_t[2] + multiply_t[2]

        grad_q_depth[0] = grad_q_depth[0] + d_depth_d_q[0, 0]
        grad_q_depth[1] = grad_q_depth[1] + d_depth_d_q[0, 1]
        grad_q_depth[2] = grad_q_depth[2] + d_depth_d_q[0, 2]
        grad_q_depth[3] = grad_q_depth[3] + d_depth_d_q[0, 3]

        grad_t_depth[0] = grad_t_depth[0] + d_depth_d_t[0,  0]
        grad_t_depth[1] = grad_t_depth[1] + d_depth_d_t[0,  1]
        grad_t_depth[2] = grad_t_depth[2] + d_depth_d_t[0,  2]
        # ------------------------------------------------------------

        d_Sigma_prime_d_q, d_Sigma_prime_d_s = gaussian_point_3d.project_to_camera_covariance_jacobian(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
            translation_camera=translation_camera,
        )

        ray_direction = gaussian_point_3d.translation - ray_origin
        _, r_jacobian, g_jacobian, b_jacobian = gaussian_point_3d.get_color_with_jacobian_by_ray(
            ray_origin=ray_origin,
            ray_direction=ray_direction,
        )
        color_r_grad = point_grad_color[0] * r_jacobian
        color_g_grad = point_grad_color[1] * g_jacobian
        color_b_grad = point_grad_color[2] * b_jacobian

        translation_grad = ti.math.vec3([0., 0., 0.])
        if enable_depth_grad:
            d_depth_d_translation = gaussian_point_3d.depth_jacobian(
                T_camera_world=T_camera_pointcloud_mat,
            )
            translation_grad = point_grad_uv @ d_uv_d_translation + \
                point_grad_depth * d_depth_d_translation
        else:
            translation_grad = point_grad_uv @ d_uv_d_translation

        # cov is Sigma
        gaussian_q_grad = point_grad_uv_cov_flat @ d_Sigma_prime_d_q
        gaussian_s_grad = point_grad_uv_cov_flat @ d_Sigma_prime_d_s

        for i in ti.static(range(3)):
            grad_pointcloud[point_id, i] = translation_grad[i]
        for i in ti.static(range(4)):
            grad_pointcloud_features[point_id, i] = gaussian_q_grad[i]
        for i in ti.static(range(3)):
            grad_pointcloud_features[point_id, i + 4] = gaussian_s_grad[i]
        for i in ti.static(range(16)):
            grad_pointcloud_features[point_id, i + 8] = color_r_grad[i]
            grad_pointcloud_features[point_id, i + 24] = color_g_grad[i]
            grad_pointcloud_features[point_id, i + 40] = color_b_grad[i]


@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row -
              1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])


@ti.func
def quaternion_to_rotation_matrix_torch_jacobian(q):
    qx, qy, qz, qw = q
    dR_dqx = mat3x3f([
        [0, 2*qy, 2*qz],
        [2*qy, -4*qx, -2*qw],
        [2*qz, 2*qw, -4*qx]
    ])
    dR_dqy = mat3x3f([
        [-4*qy, 2*qx, 2*qw],
        [2*qx, 0, 2*qz],
        [-2*qw, 2*qz, -4*qy]
    ])
    dR_dqz = mat3x3f([
        [-4*qz, -2*qw, 2*qx],
        [2*qw, -4*qz, 2*qy],
        [2*qx, 2*qy, 0]
    ])
    dR_dqw = mat3x3f([
        [0, -2*qz, 2*qy],
        [2*qz, 0, -2*qx],
        [-2*qy, 2*qx, 0]
    ])
    return dR_dqx, dR_dqy, dR_dqz, dR_dqw


@ti.func
def project_to_camera_relative_position_jacobian(
    w_t_w_points,
    T_camera_world,
    projective_transform,
):

    w_t_w_pointst_homogeneous = ti.math.vec4(
        [w_t_w_points[0], w_t_w_points[1], w_t_w_points[2], 1])
    t = T_camera_world @ w_t_w_pointst_homogeneous
    K = projective_transform

    d_uv_d_translation_camera = mat2x3f([
        [K[0, 0] / (t[2]+_EPS), K[0, 1] / (t[2]+_EPS),
            (-K[0, 0] * t[0] - K[0, 1] * t[1]) / (t[2] * t[2]+_EPS)],
        [K[1, 0] / (t[2]+_EPS), K[1, 1] / (t[2]+_EPS), (-K[1, 0] * t[0] - K[1, 1] * t[1]) / (t[2] * t[2]+_EPS)]])

    return d_uv_d_translation_camera


class GaussianPointCloudContinuousPoseRasterisation(torch.nn.Module):
    @dataclass
    class GaussianPointCloudContinuousPoseRasterisationConfig(YAMLWizard):
        near_plane: float = 0.8
        far_plane: float = 1000.
        depth_to_sort_key_scale: float = 100.
        rgb_only: bool = False
        grad_color_factor = 5.
        grad_high_order_color_factor = 1.
        grad_s_factor = 0.5
        grad_q_factor = 1.
        grad_alpha_factor = 20.
        enable_depth_grad: bool = True

    @dataclass
    class GaussianPointCloudContinuousPoseRasterisationInput:
        point_cloud: torch.Tensor  # Nx3
        point_cloud_features: torch.Tensor  # NxM
        # (N,), we allow points belong to different objects,
        # different objects may have different camera poses.
        # By moving camera, we can actually handle moving rigid objects.
        # if no moving objects, then everything belongs to the same object with id 0.
        # it shall works better once we also optimize for camera pose.
        point_object_id: torch.Tensor
        point_invalid_mask: torch.Tensor  # N
        camera_info: CameraInfo
        current_pose: torch.Tensor
        color_max_sh_band: int = 2

    @dataclass
    class BackwardValidPointHookInput:
        point_id_in_camera_list: torch.Tensor  # M
        grad_point_in_camera: torch.Tensor  # Mx3
        grad_pointfeatures_in_camera: torch.Tensor  # Mx56
        grad_viewspace: torch.Tensor  # Mx2
        magnitude_grad_viewspace: torch.Tensor  # M
        magnitude_grad_viewspace_on_image: torch.Tensor  # HxWx2
        num_overlap_tiles: torch.Tensor  # M
        num_affected_pixels: torch.Tensor  # M
        point_depth: torch.Tensor  # M
        point_uv_in_camera: torch.Tensor  # Mx2

    def __init__(
        self,
        config: GaussianPointCloudContinuousPoseRasterisationConfig,
        backward_valid_point_hook: Optional[Callable[[
            BackwardValidPointHookInput], None]] = None,
    ):
        super().__init__()
        self.config = config

        class _module_function(torch.autograd.Function):
            @staticmethod
            def forward(ctx,
                        pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        camera_info,
                        current_pose,
                        color_max_sh_band,
                        ):

                q_pointcloud_camera = current_pose[0, 3:]
                t_pointcloud_camera = current_pose[0, :3]

                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.int8, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                q_camera_pointcloud, t_camera_pointcloud = inverse_SE3_qt_torch(
                    q=q_pointcloud_camera, t=t_pointcloud_camera)
                q_camera_pointcloud = q_camera_pointcloud.unsqueeze(0)
                t_camera_pointcloud = t_camera_pointcloud.unsqueeze(0)
                # Step 1: filter points
                filter_point_in_camera(
                    pointcloud=pointcloud,
                    point_invalid_mask=point_invalid_mask,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_in_camera_mask=point_in_camera_mask,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                )
                point_in_camera_mask = point_in_camera_mask.bool()

                # Get id based on the camera_mask
                point_id_in_camera_list = point_id[point_in_camera_mask].contiguous(
                )
                del point_id
                del point_in_camera_mask

                # Number of points in camera
                num_points_in_camera = point_id_in_camera_list.shape[0]

                # Allocate memory
                point_uv = torch.empty(
                    size=(num_points_in_camera, 2), dtype=torch.float32, device=pointcloud.device)
                point_alpha_after_activation = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)
                point_in_camera = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_uv_conic = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_color = torch.zeros(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_radii = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)

                # Step 2: get 2d features
                generate_point_attributes_in_camera_plane(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    point_id_list=point_id_in_camera_list,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_conic=point_uv_conic,
                    point_alpha_after_activation=point_alpha_after_activation,
                    point_color=point_color,
                    point_radii=point_radii,
                )
                
                # Step 3: get how many tiles overlapped, in order to allocate memory
                num_overlap_tiles = torch.empty_like(point_id_in_camera_list)
                generate_num_overlap_tiles(
                    num_overlap_tiles=num_overlap_tiles,
                    point_uv=point_uv,
                    point_radii=point_radii,
                    camera_width=camera_info.camera_width,
                    camera_height=camera_info.camera_height,
                )
                # Calculate pre-sum of number_overlap_tiles
                accumulated_num_overlap_tiles = torch.cumsum(
                    num_overlap_tiles, dim=0)
                if len(accumulated_num_overlap_tiles) > 0:
                    total_num_overlap_tiles = accumulated_num_overlap_tiles[-1]
                else:
                    total_num_overlap_tiles = 0
                # The space of each point.
                accumulated_num_overlap_tiles = torch.cat(
                    (torch.zeros(size=(1,), dtype=torch.int32, device=pointcloud.device),
                     accumulated_num_overlap_tiles[:-1]))

                # del num_overlap_tiles

                # 64-bits key
                point_in_camera_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int64, device=pointcloud.device)
                # Corresponding to the original position, the record is the point offset in the frustum (engineering optimization)
                point_offset_with_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int32, device=pointcloud.device)

                # Step 4: calclualte key
                if point_in_camera_sort_key.shape[0] > 0:
                    generate_point_sort_key_by_num_overlap_tiles(
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_radii=point_radii,
                        accumulated_num_overlap_tiles=accumulated_num_overlap_tiles,  # input
                        point_offset_with_sort_key=point_offset_with_sort_key,  # output
                        point_in_camera_sort_key=point_in_camera_sort_key,  # output
                        camera_width=camera_info.camera_width,
                        camera_height=camera_info.camera_height,
                        depth_to_sort_key_scale=self.config.depth_to_sort_key_scale,
                    )

                point_in_camera_sort_key, permutation = point_in_camera_sort_key.sort()
                point_offset_with_sort_key = point_offset_with_sort_key[permutation].contiguous(
                )  # now the point_offset_with_sort_key is sorted by the sort_key
                del permutation

                tiles_per_row = camera_info.camera_width // 16
                tiles_per_col = camera_info.camera_height // 16
                tile_points_start = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                tile_points_end = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                # Find tile's start and end.
                if point_in_camera_sort_key.shape[0] > 0:
                    find_tile_start_and_end(
                        point_in_camera_sort_key=point_in_camera_sort_key,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                    )

                # Allocate space for the image.
                rasterized_image = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, 3, dtype=torch.float32, device=pointcloud.device)
                rasterized_depth = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_accumulated_alpha = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_offset_of_last_effective_point = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                pixel_valid_point_count = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                # print(f"num_points: {pointcloud.shape[0]}, num_points_in_camera: {num_points_in_camera}, num_points_rendered: {point_in_camera_sort_key.shape[0]}")

                # Step 5: render
                if point_in_camera_sort_key.shape[0] > 0:
                    # import ipdb;ipdb.set_trace()
                    gaussian_point_rasterisation(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_uv_conic=point_uv_conic,
                        point_alpha_after_activation=point_alpha_after_activation,
                        point_color=point_color,
                        rasterized_image=rasterized_image,
                        rgb_only=self.config.rgb_only,
                        rasterized_depth=rasterized_depth,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        pixel_valid_point_count=pixel_valid_point_count)
                ctx.save_for_backward(
                    pointcloud,
                    pointcloud_features,
                    # point_id_with_sort_key is sorted by tile and depth and has duplicated points, e.g. one points is belong to multiple tiles
                    point_offset_with_sort_key,
                    point_id_in_camera_list,  # point_in_camera_id does not have duplicated points
                    tile_points_start,
                    tile_points_end,
                    pixel_accumulated_alpha,
                    rasterized_depth,
                    pixel_offset_of_last_effective_point,
                    num_overlap_tiles,
                    point_object_id,
                    q_pointcloud_camera,
                    q_camera_pointcloud,
                    t_pointcloud_camera,
                    t_camera_pointcloud,
                    current_pose,
                    point_uv,
                    point_in_camera,
                    point_uv_conic,
                    point_alpha_after_activation,
                    point_color,
                )
                ctx.camera_info = camera_info
                ctx.color_max_sh_band = color_max_sh_band
                # rasterized_image.requires_grad_(True)
                return rasterized_image, rasterized_depth, pixel_valid_point_count, pixel_accumulated_alpha

            @staticmethod
            def backward(ctx, grad_rasterized_image, grad_rasterized_depth,
                         grad_pixel_valid_point_count, grad_pixel_accumulated_alpha):
                grad_pointcloud = grad_pointcloud_features = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, \
                        pointcloud_features, \
                        point_offset_with_sort_key, \
                        point_id_in_camera_list, \
                        tile_points_start, \
                        tile_points_end, \
                        pixel_accumulated_alpha, \
                        rasterized_depth, \
                        pixel_offset_of_last_effective_point, \
                        num_overlap_tiles, \
                        point_object_id, \
                        q_pointcloud_camera, \
                        q_camera_pointcloud, \
                        t_pointcloud_camera, \
                        t_camera_pointcloud, \
                        current_pose, \
                        point_uv, \
                        point_in_camera, \
                        point_uv_conic, \
                        point_alpha_after_activation, \
                        point_color = ctx.saved_tensors
                    camera_info = ctx.camera_info
                    color_max_sh_band = ctx.color_max_sh_band
                    grad_rasterized_image = grad_rasterized_image.contiguous()
                    enable_depth_grad = self.config.enable_depth_grad

                    if enable_depth_grad:
                        grad_rasterized_depth = grad_rasterized_depth.contiguous()
                        in_camera_grad_depth_buffer = torch.zeros(
                            size=(point_id_in_camera_list.shape[0], ), dtype=torch.float32, device=pointcloud.device)
                    else:  # taichi does not support None for tensor, so we use an empty tensor instead
                        grad_rasterized_depth = torch.empty(
                            size=(0, 0, ), dtype=torch.float32, device=pointcloud.device)
                        in_camera_grad_depth_buffer = torch.empty(
                            size=(0, ), dtype=torch.float32, device=pointcloud.device)
                    grad_pointcloud = torch.zeros_like(pointcloud)
                    grad_pointcloud_features = torch.zeros_like(
                        pointcloud_features)

                    grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], ), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace_on_image = torch.empty_like(
                        grad_rasterized_image[:, :, :2])

                    in_camera_grad_uv_cov_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_grad_color_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_num_affected_pixels = torch.zeros(
                        size=(point_id_in_camera_list.shape[0],), dtype=torch.int32, device=pointcloud.device)

                    grad_q = torch.zeros(
                        size=(4,), dtype=torch.float32, device=pointcloud.device)
                    grad_t = torch.zeros(
                        size=(3,), dtype=torch.float32, device=pointcloud.device)
                    grad_q_depth = torch.zeros(
                        size=(4,), dtype=torch.float32, device=pointcloud.device)
                    grad_t_depth = torch.zeros(
                        size=(3,), dtype=torch.float32, device=pointcloud.device)
                    # grad_delta_pose = torch.zeros(
                    #     size=(6,), dtype=torch.float32, device=pointcloud.device)
                    grad_q = torch.squeeze(grad_q)
                    grad_t = torch.squeeze(grad_t)
                    grad_t_depth = torch.squeeze(grad_t_depth)
                    grad_q_depth = torch.squeeze(grad_q_depth)

                    q_pointcloud_camera = q_pointcloud_camera.unsqueeze(0)
                    t_pointcloud_camera = t_pointcloud_camera.unsqueeze(0)
                    # grad_delta_pose = torch.squeeze(grad_delta_pose)
                    gaussian_point_rasterisation_backward_with_pose(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics,
                        point_object_id=point_object_id,
                        q_camera_pointcloud=q_camera_pointcloud,
                        t_camera_pointcloud=t_camera_pointcloud,
                        q_pointcloud_camera=q_pointcloud_camera.contiguous(),
                        t_pointcloud_camera=t_pointcloud_camera.contiguous(),
                        grad_q=grad_q,
                        grad_t=grad_t,
                        grad_q_depth=grad_q_depth,
                        grad_t_depth=grad_t_depth,
                        pointcloud=pointcloud,
                        pointcloud_features=pointcloud_features,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_id_in_camera_list=point_id_in_camera_list,
                        rasterized_image_grad=grad_rasterized_image,
                        enable_depth_grad=enable_depth_grad,
                        rasterized_depth_grad=grad_rasterized_depth,
                        accumulated_alpha_grad=grad_pixel_accumulated_alpha,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        rasterized_depth=rasterized_depth,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        grad_pointcloud=grad_pointcloud,
                        grad_pointcloud_features=grad_pointcloud_features,
                        grad_uv=grad_viewspace,
                        in_camera_grad_uv_cov_buffer=in_camera_grad_uv_cov_buffer,
                        in_camera_grad_color_buffer=in_camera_grad_color_buffer,
                        in_camera_grad_depth_buffer=in_camera_grad_depth_buffer,
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_uv_conic=point_uv_conic,
                        point_alpha_after_activation=point_alpha_after_activation,
                        point_color=point_color,
                        need_extra_info=True,
                        magnitude_grad_viewspace=magnitude_grad_viewspace,
                        magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                        in_camera_num_affected_pixels=in_camera_num_affected_pixels,
                    )
                    del tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point
                    grad_pointcloud_features = self._clear_grad_by_color_max_sh_band(
                        grad_pointcloud_features=grad_pointcloud_features,
                        color_max_sh_band=color_max_sh_band)
                    grad_pointcloud_features[:,
                                             :4] *= self.config.grad_q_factor
                    grad_pointcloud_features[:,
                                             4:7] *= self.config.grad_s_factor
                    grad_pointcloud_features[:,
                                             7] *= self.config.grad_alpha_factor

                    # 8, 24, 40 are the zero order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             8] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             24] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             40] *= self.config.grad_color_factor
                    # other coefficients are the higher order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             9:24] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             25:40] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             41:] *= self.config.grad_high_order_color_factor

                    if backward_valid_point_hook is not None:
                        point_id_in_camera_list = point_id_in_camera_list.contiguous().long()
                        backward_valid_point_hook_input = GaussianPointCloudContinuousPoseRasterisation.BackwardValidPointHookInput(
                            point_id_in_camera_list=point_id_in_camera_list,
                            grad_point_in_camera=grad_pointcloud[point_id_in_camera_list.long(
                            )],
                            grad_pointfeatures_in_camera=grad_pointcloud_features[
                                point_id_in_camera_list],
                            grad_viewspace=grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace=magnitude_grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                            num_overlap_tiles=num_overlap_tiles,
                            num_affected_pixels=in_camera_num_affected_pixels,
                            point_uv_in_camera=point_uv,
                            point_depth=point_in_camera[:, 2],
                        )
                        backward_valid_point_hook(
                            backward_valid_point_hook_input)
                """_summary_
                pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        q_pointcloud_camera,
                        t_pointcloud_camera,
                        camera_info,
                        color_max_sh_band,

                Returns:
                    _type_: _description_
                """

                # Convert the Python list to a PyTorch tensor

                grad_q = grad_q.view(1, 4)
                grad_t = grad_t.view(1, 3)

                grad_q_depth = grad_q_depth.view(1, 4)
                grad_t_depth = grad_t_depth.view(1, 3)

                grad_delta_pose_pointcloud_camera = torch.hstack(
                    (grad_t, grad_q)) 

                grad_delta_pose_pointcloud_camera = grad_delta_pose_pointcloud_camera.view(
                    1, 7)
                if torch.isinf(grad_delta_pose_pointcloud_camera).any():
                    print("BACKWARD - INF detected")
                    print(grad_delta_pose_pointcloud_camera)
                    print(grad_q)
                    print(grad_t)
                # DEBUG
                return grad_pointcloud, \
                    grad_pointcloud_features, \
                    None, \
                    None, \
                    None, \
                    grad_delta_pose_pointcloud_camera, \
                    None, \
                    # None pointcloud,

        self._module_function = _module_function

    def _clear_grad_by_color_max_sh_band(self, grad_pointcloud_features: torch.Tensor, color_max_sh_band: int):
        if color_max_sh_band == 0:
            grad_pointcloud_features[:, 8 + 1: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 1: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 1: 40 + 16] = 0.
        elif color_max_sh_band == 1:
            grad_pointcloud_features[:, 8 + 4: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 4: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 4: 40 + 16] = 0.
        elif color_max_sh_band == 2:
            grad_pointcloud_features[:, 8 + 9: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 9: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 9: 40 + 16] = 0.
        elif color_max_sh_band >= 3:
            pass
        return grad_pointcloud_features

    def forward(self, input_data: GaussianPointCloudContinuousPoseRasterisationInput):
        pointcloud = input_data.point_cloud
        pointcloud_features = input_data.point_cloud_features
        point_invalid_mask = input_data.point_invalid_mask
        point_object_id = input_data.point_object_id
        current_pose = input_data.current_pose
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % 16 == 0
        assert camera_info.camera_height % 16 == 0
        return self._module_function.apply(
            pointcloud,
            pointcloud_features,
            point_invalid_mask,
            point_object_id,      
            camera_info,
            current_pose,
            color_max_sh_band,
        )
