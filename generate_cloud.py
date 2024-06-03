import cv2
import open3d as o3d
from PIL import Image
import numpy as np

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

# # 加载RGB图像和深度图像
# rgb_image = cv2.imread('c:/Users/LENOVO/Desktop/color_0.png')
# depth_image = cv2.imread('c:/Users/LENOVO/Desktop/depth_0.png', cv2.IMREAD_ANYDEPTH)

color = np.array(Image.open('doc/bottle/bottle.png'), dtype=np.float32) / 255.0
depth = np.array(Image.open('doc/bottle/depth.png'))
# workspace_mask = np.array(Image.open('/home/hongrui/graspnet-baseline/doc/bottle/mask.npy'))
# workspace_mask = np.load('/home/hongrui/graspnet-baseline/doc/bottle/mask.npy')
# intrinsic = np.array([609.765, 0.0, 322.594, 0.0, 608.391, 243.647, 0.0, 0.0, 1.0]).reshape((3, 3)) # 609.765, 0.0, 322.594, 0.0, 608.391, 243.647, 0.0, 0.0, 1.0
intrinsic = np.array([[637.91, 0., 639.65],
                        [0., 637.91, 391.311],
                        [0., 0., 1.]])
factor_depth = np.array([[1000.]])

camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)  # [480,270]  [1280,720]
cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

cloud_nomask = cloud
color_nomask = color
cloud_nomask = cloud_nomask[depth > 0]
color_nomask = color_nomask[depth > 0]

cloud = o3d.geometry.PointCloud()
print(cloud_nomask.shape)
print(color_nomask.shape)
# cloud_nomask = cloud_nomask[depth > 0]
# color_nomask = color_nomask[depth > 0]
# cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
# cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))

# 可视化点云
o3d.visualization.draw_geometries([cloud])

# # 创建点云对象
# point_cloud = o3d.geometry.PointCloud()

# # 获取相机内参
# fx = 500  # 深度图像的焦距
# fy = 500  # 深度图像的焦距
# cx = rgb_image.shape[1] / 2  # 图像的中心点
# cy = rgb_image.shape[0] / 2  # 图像的中心点

# # 生成点云
# rows, cols = depth_image.shape
# for y in range(rows):
#     for x in range(cols):
#         depth = depth_image[y, x]
#         if depth == 0:
#             continue
#         z = depth / 1000.0  # 将深度从毫米转换为米
#         x = (x - cx) * z / fx
#         y = (y - cy) * z / fy
#         point_cloud.points.append([x, y, z])

# # 设置点云的颜色
# colors = rgb_image.reshape(-1, 3)
# point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

# # 可视化点云
# o3d.visualization.draw_geometries([point_cloud])