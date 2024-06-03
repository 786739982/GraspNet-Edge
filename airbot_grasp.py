import airbot
import time
import argparse
from GraspPosePre import GraspPosePrediction
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import numpy as np
from PIL import Image
import pyrealsense2 as rs

from utils.quaternion_utils import pose_transform
from utils.merge_cloud import merge_clouds
from utils.generate_cloud import create_cloud
from utils.get_image import get_color_depth

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

'''
    Init airbot.
'''

# TODO：modify the path to the airbot_play urdf file
urdf_path = (
    "/home/ghz/Work/airbot_play/arm-control/models/airbot_play_v2_1/urdf/airbot_play_v2_1_with_gripper.urdf"
)

# specify the fk/ik/dk classes to use
fk = airbot.AnalyticFKSolver(urdf_path)
ik = airbot.ChainIKSolver(urdf_path)
id = airbot.ChainIDSolver(urdf_path, "down")
# instance the airbot player
airbot_player = airbot.create_agent(fk, ik, id, "vcan0", 1.0, "gripper", False, False)

# wait for the robot move to the initial zero pose
time.sleep(2)
# get current joint positions(q), velocities(v) and torques(t)
# all are six-elements tuple containing current values of joint1-6
cp = airbot_player.get_current_joint_q()
cv = airbot_player.get_current_joint_v()
ct = airbot_player.get_current_joint_t()
print(cp)




'''
    Get point cloud.
'''

# TODO : modify to dynamic image data

# Init paramter of camera
intrinsic = np.array([[637.91, 0., 639.65],
                        [0., 637.91, 391.311],
                        [0., 0., 1.]])
factor_depth = np.array([[1.]])

# Init RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
profile = pipeline.start(config)
color, depth = get_color_depth(pipeline, profile, save=True)
color = np.flip(color/255.0, axis=-1)
print(color)
source_cloud = create_cloud(color, depth, intrinsic, factor_depth)
# o3d.visualization.draw([source_cloud])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
profile = pipeline.start(config)
color, depth = get_color_depth(pipeline , profile, save=True)
color = np.flip(color/255.0, axis=-1)
print(color)
target_cloud = create_cloud(color, depth, intrinsic, factor_depth)
# o3d.visualization.draw([target_cloud])

print("Load two point clouds and show initial pose ...")
merge_cloud = merge_clouds(target_cloud, source_cloud)



'''
    Predict grasp pose
'''

# Init grasp prediction
grasper = GraspPosePrediction(checkpoint_path=cfgs.checkpoint_path, num_point=cfgs.num_point, num_view=cfgs.num_view,
                              collision_thresh=cfgs.collision_thresh, voxel_size=cfgs.voxel_size,
                              camera_width=1280.0, camera_high=720.0)

translation, rotation = grasper.get_grasp(merge_cloud, np.ones((720, 1280))) # TODO : get mask of object




'''
    Pose translation and set
'''

end_rotation = np.array([   [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1] ])
# TODO: modifiy real var of Claw , not base!
end_translation = np.array([0.01, 0, 0.065])  

# Hand-eye coordinate transformation 
# TODO: modyfi urdf
rotation, translation = pose_transform(rotation, translation, end_rotation, end_translation)
rotation, translation = pose_transform(rotation, translation, 
                                       airbot_player.get_current_rotation(), airbot_player.get_current_translation())

airbot_player.set_target_translation(translation)
airbot_player.set_target_rotation(rotation)
