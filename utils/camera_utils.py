import os

import numpy as np
from PIL import Image
import torch
import torchvision

from utils.common import get_json_content
from utils.pose_utils import PoseFlow


# 将 PIL 图像转换为归一化的 PyTorch 张量
def get_normalized_torch_image(image: Image):
    # 转换管道
    torch_convert = torchvision.transforms.Compose(
        # 将 PIL 图像转换为 PyTorch 张量
        [torchvision.transforms.ToTensor(),
        # 对张量进行归一化处理   ImageNet 数据集的常用归一化参数：均值       标准差
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch_convert(np.array(image)).unsqueeze(0)  # 最前面增加一个维度，通常用于表示批次大小。这使得返回的张量形状为 (1, C, H, W)，适合于模型输入

# 将 NumPy 数组转换为 PyTorch 张量
def get_torch_intrinsics_or_extrinsics(intrinsic: np.array):
    return torch.from_numpy(intrinsic).float().unsqueeze(0) 


# 解析相机信息的类，返回四个相机的内参和外参
class CameraInfoParser:
    # 初始化
    def __init__(self, task_index, parser_dir):
        self.task_index = task_index
        self.parser_dir = parser_dir    # "./utils"
        # 解析相机信息，并返回内参和外参
        self.intrinsic, self.extrinsic = self.parser_info()

    # 解析相机的内参和外参
    def parser_info(self):
        # 读取存储在 parser_dir(./utils) 目录下的 camera_config_right_hand.json 文件
        # camera_config_right_hand.json 文件包含四个相机的外参和内参
        camera_info = get_json_content(os.path.join(self.parser_dir, "camera_config_right_hand.json"))
        camera_info_right_hand = {} # 相机信息字典

        # 遍历 camera_info 中的每个相机通道（camera_channel）及其对应的信息（cam_info）
        for camera_channel, cam_info in camera_info.items():
            # 获取相机标签(rgb_front,rgb_left,rgb_right,rgb_rear)
            camera_label = self.get_dir_name_from_channel(camera_channel)
            # 存储相机的外参（位置和姿态）以及内参（宽度、高度、视场角）
            camera_info_right_hand[camera_label] = {
                "x": cam_info["extrinsics"]["x"], "y": cam_info["extrinsics"]["y"], "z": cam_info["extrinsics"]["z"],
                "roll": cam_info["extrinsics"]["roll"], "pitch": cam_info["extrinsics"]["pitch"], "yaw": cam_info["extrinsics"]["yaw"], 
                "width": cam_info["intrinsics"]["width"], "height": cam_info["intrinsics"]["height"], "fov": cam_info["intrinsics"]["fov"]
            }

        # 创建两个空字典，分别用于存储每个相机的内参和外参
        intrinsic = {}
        extrinsic = {}
        # 遍历 camera_info_right_hand 中的每个相机标签（cam_label）及其规格（cam_spec）
        # 分别将内参和外参传入到字典中
        for cam_label, cam_spec in camera_info_right_hand.items():
            intrinsic[cam_label] = self.get_intrinsics(cam_spec["width"], cam_spec["height"], cam_spec["fov"])
            extrinsic[cam_label] = self.get_extrinsics(cam_spec["x"], cam_spec["y"], cam_spec["z"], cam_spec["yaw"], cam_spec["pitch"], cam_spec["roll"])
        return intrinsic, extrinsic

    # 计算相机的外参矩阵
    def get_extrinsics(self, x_right, y_right, z_right, yaw_right, pitch_right, roll_right):
        # 相机到像素的转换矩阵
        cam2pixel_3 = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]], dtype=float)
        # 计算旋转矩阵
        pose_flow_obj = PoseFlow(att_input=[yaw_right, pitch_right, roll_right], type="euler", deg_or_rad="deg")
        rotation_3 = pose_flow_obj.get_rotation_matrix()
        # 计算平移向量
        translation = np.array([[x_right], [y_right], [z_right]])
        # 将旋转矩阵和平移向量合并成一个 4x4 的外参矩阵
        cam2veh_3 = self.concat_rotaion_and_translation(rotation_3, translation).tolist()
        # 计算从车辆坐标系到相机坐标系的转换
        veh2cam = cam2pixel_3 @ np.array(np.linalg.inv(cam2veh_3))
        return veh2cam

    # 计算相机的内参矩阵
    def get_intrinsics(self, width, height, fov):
        f = width / (2 * np.tan(np.deg2rad(fov) / 2))  # 焦距
        Cu = width / 2      # 图像的主点（光轴与图像平面的交点）坐标
        Cv = height / 2
        # 内参矩阵
        intrinsic = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        return intrinsic

    def concat_rotaion_and_translation(self, rotation, translation):
        ret = np.eye(4)
        ret[:3, :] = np.concatenate([rotation, translation], axis=1)
        return ret

    # 根据相机通道名称生成一个标准化的目录名称
    # [例子] 输入：CAM_FRONT   输出：rgb_front
    def get_dir_name_from_channel(self, channel_name):
        return "_".join(['rgb'] + channel_name.split("_")[1:]).lower()



class ProcessImage:
    # 初始化
    def __init__(self, pil_image: Image, intrinsics, extrinsics, target_size=(256, 256)):
        self.pil_image = pil_image
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        # 原始图像的宽度和高度
        self.image_width, self.image_height = self.pil_image.size
        # 目标尺寸
        self.target_width, self.target_height = target_size
        self.width_scale = self.target_width / self.image_width
        # 缩放因子
        self.height_scale = self.target_height / self.image_height

        self.resize_img = None
        self.resize_intrinsics = None

    # 调整图像大小和内参
    def resize_pil_image(self):
        # 使用最近邻插值将图像调整为目标尺寸
        self.resize_img = self.pil_image.resize((self.target_width, self.target_height), resample=Image.NEAREST)
        # 更新内参
        self.resize_intrinsics = self._update_intrinsics()
        return self.resize_img, self.resize_intrinsics

    # 根据图像的缩放比例更新相机的内参矩阵
    def _update_intrinsics(self):
        updated_intrinsics = self.intrinsics.copy()
        # Adjust intrinsics scale due to resizing
        updated_intrinsics[0, 0] *= self.width_scale    # 焦距（fx），反映图像宽度的变化
        updated_intrinsics[0, 2] *= self.width_scale    # 主点 X 坐标，反映图像宽度的变化
        updated_intrinsics[1, 1] *= self.height_scale   # 焦距（fy），反映图像高度的变化
        updated_intrinsics[1, 2] *= self.height_scale   # 主点 Y 坐标，反映图像高度的变化

        return updated_intrinsics