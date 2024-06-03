import open3d as o3d
import numpy as np

def merge_pointclouds(rgb_images, depth_images, intrinsics, extrinsics):
    # 创建一个空的点云对象
    merged_pc = o3d.geometry.PointCloud()

    # 遍历每个视角的RGB-D图像
    for rgb_image, depth_image, intrinsic, extrinsic in zip(rgb_images, depth_images, intrinsics, extrinsics):
        # 将RGB图像和深度图像转换为点云
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_trunc=3.0, convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # 将点云从相机坐标系转换到共享的参考坐标系
        pc.transform(extrinsic)

        # 将当前视角的点云合并到整体点云中
        merged_pc += pc

    return merged_pc



if __name__ == '__main__':

    # 示例数据
    rgb_images = [o3d.io.read_image("rgb_image1.png"), o3d.io.read_image("rgb_image2.png")]
    depth_images = [o3d.io.read_image("depth_image1.png"), o3d.io.read_image("depth_image2.png")]
    intrinsics = [o3d.camera.PinholeCameraIntrinsic(), o3d.camera.PinholeCameraIntrinsic()]
    extrinsics = [np.eye(4), np.eye(4)]  # 假设外参矩阵为单位矩阵

    # 合并点云
    merged_pointcloud = merge_pointclouds(rgb_images, depth_images, intrinsics, extrinsics)

    # 可选：对合并后的点云进行滤波或其他后处理操作
    merged_pointcloud = merged_pointcloud.voxel_down_sample(voxel_size=0.05)  # 使用体素下采样滤波

    # 可视化点云
    o3d.visualization.draw_geometries([merged_pointcloud])