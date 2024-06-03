# import open3d as o3d

# # 创建点云对象
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

# # 创建窗口并可视化点云
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='Open3D')
# vis.add_geometry(point_cloud)
# vis.update_geometry(point_cloud)
# vis.poll_events()
# vis.update_renderer()

# # 截图并保存为图像文件
# image = vis.capture_screen_float_buffer(do_render=True)
# o3d.io.write_image("point_cloud.png", image)

# # 关闭窗口
# vis.destroy_window()

import open3d as o3d
import numpy as np

# 创建一个包含三维坐标的Python列表
points_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

# 使用Vector3dVector函数将Python列表转换为Open3D的点云数据类型
points = o3d.utility.Vector3dVector(points_list)

# 打印转换后的结果
print(points)