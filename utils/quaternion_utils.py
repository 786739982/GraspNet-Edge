""" Rotation_matrix to Quaternion.
    Author: Hongrui-Zhu
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rota_to_quat(rotation_matrix):

    # 将旋转矩阵转换为四元数
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()

    # 计算旋转四元数和 (1, 0, 0) 之间的差异
    target_rotation = Rotation.from_rotvec(np.pi / 2 * np.array([1, 0, 0]))
    desired_quaternion = r * target_rotation
    desired_quaternion = desired_quaternion.as_quat()
    # print(desired_quaternion)
    
    return desired_quaternion

def pose_transform(rotation_camera, translation_camera, hand_to_robot_rotation, hand_to_robot_translation):
    # 将相机坐标系下的旋转矩阵和平移向量转换为机械臂坐标系下的旋转矩阵和平移向量

    # 转换旋转矩阵
    rotation_robot = np.dot(hand_to_robot_rotation, rotation_camera)

    # 转换平移向量
    translation_robot = np.dot(hand_to_robot_rotation, translation_camera) + hand_to_robot_translation

    return rotation_robot, translation_robot

def is_approximately_equal(target_translation, target_rotation, current_translation, current_rotation, translation_threshold, rotation_threshold):
    # 判断平移是否近似相等
    translation_difference = np.linalg.norm(target_translation - current_translation)
    translation_equal = translation_difference < translation_threshold

    # 判断旋转是否近似相等
    rotation_difference = np.linalg.norm(target_rotation - current_rotation)
    rotation_equal = rotation_difference < rotation_threshold

    # 返回平移和旋转是否近似相等的判断结果
    return translation_equal and rotation_equal

if __name__=='__main__':
    # 输入相机坐标系下的旋转矩阵和平移向量，以及手眼坐标系的变换关系
    rotation_camera = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    translation_camera = np.array([0.1, 0.2, 0.3])
    hand_to_robot_rotation = np.array ([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])
    hand_to_robot_translation = np.array([0.4, 0.5, 0.6])

    # 调用函数进行坐标系转换
    rotation_robot, translation_robot = pose_transform(rotation_camera, translation_camera, hand_to_robot_rotation, hand_to_robot_translation)

    # 输出机械臂坐标系下的旋转矩阵和平移向量
    print("Rotation matrix (Robot):")
    print(rotation_robot)
    print("Translation (Robot):")
    print(translation_robot)
