# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""ICP variant that uses both geometry and color for registration"""

import open3d as o3d
import numpy as np
import copy
from PIL import Image
from generate_cloud import create_cloud

def registration_result(source, target, transformation, visual=False):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    if visual:
        o3d.visualization.draw([source_temp,target])
    return source_temp

def merge_clouds(source, target, visual=False):
    current_transformation = np.identity(4)
    # registration_result(source, target, current_transformation)
    print(current_transformation)

    # Colored pointcloud registration.
    # This is implementation of following paper:
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017.
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("Colored point cloud registration ...\n")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("2. Estimate normal")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp, "\n")
    print(current_transformation)
    return registration_result(source, target, result_icp.transformation, visual)
    

if __name__ == "__main__":
    print("Load two point clouds and show initial pose ...")

    color_s = np.array(Image.open('c:/Users/LENOVO/Desktop/image/color_0.png'), dtype=np.float32) / 255.0
    depth_s = np.array(Image.open('c:/Users/LENOVO/Desktop/image/depth_0.png'), dtype=np.float32)
    color_t = np.array(Image.open('c:/Users/LENOVO/Desktop/image/color_1.png'), dtype=np.float32) / 255.0
    depth_t = np.array(Image.open('c:/Users/LENOVO/Desktop/image/depth_1.png'), dtype=np.float32)

    intrinsic = np.array([[637.91, 0., 639.65],
                            [0., 637.91, 391.311],
                            [0., 0., 1.]])
    factor_depth = np.array([[1.]])

    source = create_cloud(color_s, depth_s, intrinsic, factor_depth)
    target = create_cloud(color_t, depth_t, intrinsic, factor_depth)
    
    merge_cloud = merge_clouds(source, target)
