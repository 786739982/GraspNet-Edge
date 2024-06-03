import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def depth_to_pointcloud(depth_image, intrinsic):
    # Create Open3D Image from depth map
    o3d_depth = o3d.geometry.Image(depth_image)

    # Get intrinsic parameters
    fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

    # Create Open3D PinholeCameraIntrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)

    # Create Open3D PointCloud object from depth image and intrinsic parameters
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

    return pcd

def save_pointcloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)



def get_color_depth(pipeline ,profile, skip_frames=20, save=False, save_path='./'):
    
    # 得到深度传感器实例
    depth_sensor = profile.get_device().first_depth_sensor()

    # 设置深度传感器参数
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = 1.0

    # 创建一个numpy数组来存储深度图像数据
    depth_image = np.zeros((480, 640), dtype=np.float32)

    align_to = rs.stream.color
    align = rs.align(align_to)
    print('align')

    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    print('depth_intrinsics')
    counter = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_depth_frame:
                continue
                
            depth_frame = frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame:
                continue

            depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data * depth_scale
            depth_image[depth_image > clipping_distance] = 0
            color_image = np.asanyarray(color_frame.get_data())

            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrinsics  = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            # pc = depth_to_pointcloud(depth_image, depth_intrinsics)

            # cv2.imshow('RealSense', color_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
            # cv2.imshow('depth_color', depth_colormap)
            
            if counter == skip_frames:
                if save:
                    # 保存 RGB 图像
                    rgb_file_path = save_path + 'color_t.png'
                    cv2.imwrite(rgb_file_path, color_image)
                    print('color saved', rgb_file_path)

                    # 保存深度图像
                    depth_file_path = save_path + 'depth_t.png'
                    cv2.imwrite(depth_file_path, depth_image*255.0)
                    # print(depth_image.shape)
                    # print(depth_image)
                    print('depth saved', depth_file_path)
                return color_image, depth_image*255.0
            else: 
                print(counter)
                counter += 1
                
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # 初始化 RealSense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    profile = pipeline.start(config)
    get_color_depth(pipeline , profile, save=True)