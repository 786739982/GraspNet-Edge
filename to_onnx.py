import torch
import sys
import os
import numpy as np
from PIL import Image
import open3d as o3d
import IPython
import scipy.io as scio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import functools

def get_and_process_data():

    color = np.array(Image.open('doc/example_data/color.png'), dtype=np.float32) / 255.0
    depth = np.array(Image.open('doc/example_data/depth.png'))
    workspace_mask = np.array(Image.open('doc/example_data/workspace_mask.png'))
    meta = scio.loadmat('doc/example_data/meta.mat')
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)  # [480,270]  [1280,720]
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_nomask = cloud
    color_nomask = color
    cloud_nomask = cloud_nomask[depth > 0]
    color_nomask = color_nomask[depth > 0]

    # sample point, accelerate to convert to onnx
    idxs_sample = np.random.choice(len(cloud_nomask), 50000, replace=False)
    cloud_nomask = cloud_nomask[idxs_sample]
    color_nomask = color_nomask[idxs_sample]

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= 20000:
        idxs = np.random.choice(len(cloud_masked), 8000, replace=False) # 20000
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), 20000-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    color_sampled = torch.from_numpy(color_sampled)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    color_sampled = color_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    #end_points['cloud_colors'] = color_sampled

    return end_points, cloud

# 加载 GraspNet 模型
model = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load checkpoint
checkpoint = torch.load('checkpoint-rs.tar', map_location=torch.device('cuda:0'))
model.load_state_dict(checkpoint['model_state_dict'])

# torch.save(model.state_dict(), 'grasp.pth')
model.eval()

end_points, cloud = get_and_process_data()
end_points_dict = {'end_points': end_points}
# print(type(end_points), type(end_points['point_clouds']))
# 进行推理
output = model(end_points)
# 打印输出的形状
print("Output keys:", output.keys())

torch.onnx.register_custom_op_symbolic('aten::lift_fresh', lambda g, x: x, 11)
with torch.no_grad():
    torch.onnx.export(model, end_points_dict, "grasp_horizon.onnx", 
                      verbose=True, do_constant_folding=True, 
                      export_params=True, opset_version=11)  #  input_names=input_names, output_names=output_names

import onnx
model_file = 'grasp_horizon.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
