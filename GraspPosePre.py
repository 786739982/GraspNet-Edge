""" Predicting grasp pose from the input of point_cloud.
    Author: Hongrui-Zhu
    Note: 
        Pay attention to modifying camera parameters("self.camera_width"  "self.camera_high" "self.intrinsic" "self.factor_depth") to adapt to hardware
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from typing import Optional

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from quaternion_utils import rota_to_quat

class GraspPosePrediction:

    def __init__(self,checkpoint_path='checkpoint-rs.tar', num_point=20000, 
                  num_view=300, collision_thresh=0.01, voxel_size=0.01, 
                  camera_width=1280.0, camera_high=720.0) -> None:
        """Initialize the model and load checkpoints (if needed)
        """
        super.__init__()
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.camera_width = camera_width
        self.camera_high = camera_high
        self.intrinsic = np.array([609.765, 0.0, 322.594, 0.0, 608.391, 243.647, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.factor_depth = np.array([[1000.]])
        self.net = self.get_net()


    def get_grasp(self, point_cloud: np.ndarray, workspace_mask): #prompt: Optional[str] = None
        """Obtain the target grasp pose given a point cloud
        Args:
            point_cloud: the point cloud array with shape (N, 3) and dtype np.float64
                Each of the N entry is the cartesian coordinates in the world base
            workspace_mask: mask for grasping targets

        Returns:
            tuple(np.ndarray, np.ndarray): the 6-DoF grasping pose in the world base 
                the first element: an np.ndarray with the size of [3] and dtype of np.float64, representing the target position of the end effector
                the second element: an np.ndarray with the size of [4] and dtype of np.float64, representing the orientation of the pose
                    this element is a quarternion representing the rotation between the target pose and (1,0,0) (pointing forward, the default orientation of end effector)

        Notes: the axis direction of the world base are:
            x -> forward
            y -> left
            z -> upward
        """
        end_points, cloud = self.get_and_process_data(point_cloud, workspace_mask)
        gg = self.get_grasps(self.net, end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        gg.nms()
        gg.sort_by_score()
        self.vis_grasps(gg, cloud)
        
        # # quaternion = rota_to_quat(rotation)
        # # translation[3], quaternion[4]
        return gg[0].translation, gg[0].rotation_matrix
    
    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    
    def get_and_process_data(self, input_cloud, workspace_mask):
        # load data
        intrinsic = self.intrinsic
        factor_depth = self.factor_depth

        # generate cloud 
        camera = CameraInfo(self.camera_width, self.camera_high, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)  # [480,270]  [1280,720]
        # cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud = input_cloud[:, :, 0:3]
        color = input_cloud[:, :, 3:]
        cloud_nomask = cloud
        color_nomask = color

        # get valid points
        mask = workspace_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud):
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])


if __name__=='__main__':
    pass