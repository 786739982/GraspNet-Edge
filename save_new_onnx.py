import torch
import sys
import os
import numpy as np
from PIL import Image
import open3d as o3d
import IPython
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, GraspNet_New
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class MyModel_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=1)
        with torch.no_grad():
            self.conv1.weight.fill_(1)
        print(self.conv1.weight)
    def forward(self, x):
        x = self.conv1(x)
        return x

def get_and_process_data():

    color = np.array(Image.open('/home/hongrui/graspnet-baseline/doc/bottle/bottle.png'), dtype=np.float32) / 255.0
    depth = np.array(Image.open('/home/hongrui/graspnet-baseline/doc/bottle/depth.png'))
    workspace_mask = np.load('/home/hongrui/graspnet-baseline/doc/bottle/mask.npy')
    # intrinsic = np.array([609.765, 0.0, 322.594, 0.0, 608.391, 243.647, 0.0, 0.0, 1.0]).reshape((3, 3)) # 609.765, 0.0, 322.594, 0.0, 608.391, 243.647, 0.0, 0.0, 1.0
    intrinsic = np.array([[637.91, 0., 639.65],
                          [0., 637.91, 391.311],
                          [0., 0., 1.]])
    factor_depth = np.array([[1000.]])

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)  # [480,270]  [1280,720]
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_nomask = cloud
    color_nomask = color
    cloud_nomask = cloud_nomask[depth > 0]
    color_nomask = color_nomask[depth > 0]

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= 20000:
        idxs = np.random.choice(len(cloud_masked), 20000, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), 20000-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    color_sampled = torch.from_numpy(color_sampled)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    color_sampled = color_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

# 加载 GraspNet 模型
model = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
modelNew = GraspNet_New(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                    cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
modelNew.to(device)

# Load checkpoint
checkpoint = torch.load('logs/checkpoint_dist.tar', map_location=torch.device('cuda:0'))
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

pretrain_dict = checkpoint['model_state_dict']

modelNew_dict = modelNew.state_dict()
pretrain_dict = {key:value for key,value in pretrain_dict.items() if key in modelNew_dict}
modelNew_dict.update(pretrain_dict)
modelNew.load_state_dict(modelNew_dict)
# print(modelNew.state_dict().keys())
# print(modelNew.state_dict()['conv.bias'])

end_points, cloud = get_and_process_data()

# 进行推理
output = modelNew(end_points)
# 打印输出的形状
print("Output shape:", output.keys())
end_points_dict = {'end_points': output}
input_names = ['point_clouds', 'cloud_colors']
output_names = [key for key in output.keys()]
print(output_names)
print([name for name, _ in modelNew.named_parameters() if _.requires_grad])
# import pdb; pdb.set_trace()
# print(type(end_points), type(end_points['point_clouds']))

torch.onnx.export(modelNew, end_points_dict, "graspnet.onnx", input_names=input_names, output_names=output_names, export_params=True, opset_version=11)
