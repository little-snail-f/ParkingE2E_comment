import os
from typing import List

import numpy as np
import torch
from shapely.geometry import LineString
from shapely.measurement import hausdorff_distance

from utils.common import get_json_content
from utils.pose_utils import CustomizePose



# 将轨迹点的坐标和进度转换为标记（token）
def tokenize_traj_point(x, y, progress, token_nums, xy_max, progress_bar=1):
    """
    Tokenize trajectory points
    :param x: [-xy_max, xy_max]
    :param y: [-xy_max, xy_max]
    :param progress: [-progress_bar, progress_bar]
    :return: tokenized control range [0, token_nums]
    """
    valid_token = token_nums - 1
    # 将 x、y 和 progress 归一化到 [0, 1] 的范围
    x_normalize = (x + xy_max) / (2 * xy_max)
    y_normalize = (y + xy_max) / (2 * xy_max)
    progress_normalize = (progress + progress_bar) / (2 * progress_bar)
   
    if x_normalize > 1 or y_normalize > 1 or progress_normalize > 1 or x_normalize < 0 or y_normalize < 0 or progress_normalize < 0:
        raise ValueError("x_normalize: {}, y_normalize: {}, progress_normalize: {}".format(x_normalize, y_normalize, progress_normalize))
    # 将归一化后的值乘以有效标记数量并转换为整数，返回标记列表
    return [int(x_normalize * valid_token), int(y_normalize * valid_token), int(progress_normalize * valid_token)]


# 将 token 形式的轨迹点转换为实际的坐标值
def detokenize_traj_point(torch_list: torch.tensor, token_nums, item_num, xy_max=10, progress_max=1):
    # 有效的标记数量
    valid_token = token_nums - 1
    # 调整输入张量形状为(batch_size, item_num)   2
    torch_list_process = torch_list.view(-1, item_num)
    # 初始化返回张量
    ret_tensor = torch.zeros_like(torch_list_process, dtype=torch.float32)
    # 计算坐标值
    ret_tensor[:, :2] = (torch_list_process[:, :2] / valid_token) * 2 * xy_max - xy_max
    if (item_num > 2):
        ret_tensor[:, 2:] = (torch_list_process[:, 2:] / valid_token) * 2 * progress_max - progress_max
    return ret_tensor


class TrajectoryInfoParser:
    def __init__(self, task_index, task_path):
        self.task_index = task_index
        self.task_path = task_path
        self.total_frames = self._get_trajectory_num()  # 轨迹帧数  247
        self.trajectory_list = self.make_trajectory()   # 轨迹列表，包含每一帧的位置信息和姿态信息
        self.progress_list = self.get_progress_list()   # 进度列表 [1.0, 0.9989255170385136, ...]
        self.candidate_target_pose = self.get_candidate_target_pose()   # 候选目标姿态

    # 获取与当前任务相关的轨迹帧数
    def _get_trajectory_num(self) -> int:
        return len(os.listdir(os.path.join(self.task_path, "measurements"))) # 计算指定目录中文件的数量
    
    def get_trajectory_point(self, point_index) -> CustomizePose:
        return self.trajectory_list[point_index]

    def get_progress(self, index) -> float:
        return self.progress_list[index]

    # 计算轨迹的方向
    def _get_trajectory_direction(self, bias_threshold=30) -> str:
        direction = None
        # 从轨迹的起始点到结束点的偏航角差
        delta_yaw = self.get_safe_yaw(self.trajectory_list[-1].yaw - self.trajectory_list[0].yaw)
        # delta_yaw 在 60 ～ 120 范围内
        if 90 - bias_threshold < abs(delta_yaw) < 90 + bias_threshold:
            direction = "right" if delta_yaw > 0 else "left"
        else:
            raise ValueError(f"Don't support trajectory rotation angle '{delta_yaw}'!")
        return direction

    # 筛选候选目标姿态
    def get_candidate_target_pose(self) -> List[CustomizePose]:
        candidate_target_pose = []
        # 遍历轨迹列表，将与最后一个轨迹点的偏航角相差小于 1 弧度的点都当作目标点
        for trajectory_item in self.trajectory_list:
            if abs(trajectory_item.yaw - self.trajectory_list[-1].yaw < 1):
                candidate_target_pose.append(trajectory_item)
        return candidate_target_pose

    def get_random_candidate_target_pose(self) -> CustomizePose:
        candidate_target_pose_len = len(self.candidate_target_pose)
        candidate_target_pose = self.candidate_target_pose[np.random.choice(range(0, candidate_target_pose_len))]
        # noise = 0.4 * (2 * np.random.rand(*candidate_target_pose.shape) - 1) # 扰动从[-0.4m, 0.4m]
        # candidate_target_pose += noise
        return candidate_target_pose
    
    def get_precise_target_pose(self) -> CustomizePose:
        return  self.trajectory_list[-1]
    
    def get_safe_yaw(self, yaw) -> int:
        if yaw <= -180:
            yaw += 360
        if yaw > 180:
            yaw -= 360
        return yaw

    # 生成并返回指定索引的 JSON 文件路径
    def get_measurement_path(self, measurement_index) -> str:
        return os.path.join(self.task_path, "measurements", "{}.json".format(str(measurement_index).zfill(4)))

    # 生成一个轨迹列表，包含每一帧的位置信息和姿态信息
    def make_trajectory(self) -> List[CustomizePose]:
        trajectory_list = []
        # 帧迭代
        for frame in range(0, self.total_frames):
            # 获取当前帧的数据
            data = get_json_content(self.get_measurement_path(frame))
            # 创建当前帧的姿态对象
            cur_pose = CustomizePose(x=data["x"], y=data["y"], z=data["z"], roll=data["roll"], yaw=data["yaw"], pitch=data["pitch"])
            trajectory_list.append(cur_pose)
        return trajectory_list

    # 计算并返回一个表示进度的列表, 用于表示在轨迹中的相对位置
    def get_progress_list(self) -> List[float]:
        distance_list = [0.0]
        # distance_list 的每个元素表示从起始位置到当前帧的总距离
        for index in range(1, self.total_frames):
            distance_list.append(distance_list[-1] + self._get_backwark_delta_distance(index))
        # 进度列表，1 - 当前距离/总距离
        progress_list = 1 - np.array(distance_list) / distance_list[-1]
        # 如果轨迹方向是 "left"，则将进度列表取反 ？？？？？
        if self._get_trajectory_direction() == "left":
            progress_list = -progress_list
        return progress_list.tolist()
    
    # 计算在给定索引处的轨迹点与前一个轨迹点之间的直线距离
    def _get_backwark_delta_distance(self, index) -> float:
        delta_y = self.get_trajectory_point(index).y - self.get_trajectory_point(index - 1).y
        delta_x = self.get_trajectory_point(index).x - self.get_trajectory_point(index - 1).x
        return np.linalg.norm([delta_x, delta_y])


class TrajectoryDistance:
    def __init__(self, prediction_points_np, gt_points_np):
        self.prediction_points_np = prediction_points_np
        self.gt_points_np = gt_points_np

        self.cut_stop_segment()


    def cut_stop_segment(self, stop_threshold=0.001):
        distance_list = np.linalg.norm(self.gt_points_np[1:, :] - self.gt_points_np[:-1, :], axis=-1)

        threshold_bool_list = abs(distance_list) < stop_threshold

    
        stop_index = -1
        for index in range(0, len(threshold_bool_list)):
            inverse_index = len(threshold_bool_list) - index - 1
            if not threshold_bool_list[inverse_index]:
                stop_index = inverse_index + 1
                break
        self.prediction_points_np = self.prediction_points_np[:stop_index + 1]
        self.gt_points_np = self.gt_points_np[:stop_index + 1]


    def get_len(self):
        return self.gt_points_np.shape[0]

    def get_l2_distance(self):
        l2_distance_list = np.linalg.norm(self.gt_points_np - self.prediction_points_np, axis=1)
        l2_distance = np.mean(l2_distance_list)
        return l2_distance


    def get_haus_distance(self):
        line_gt = LineString(self.gt_points_np)
        line_pred = LineString(self.prediction_points_np)
        haus_distance = hausdorff_distance(line_pred, line_gt)
        return haus_distance


    def get_fourier_difference(self):
        fd1 = self.compute_fourier_descriptor(self.gt_points_np, num_descriptors = 10)
        fd2 = self.compute_fourier_descriptor(self.prediction_points_np, num_descriptors = 10)
        fourier_difference = np.linalg.norm(fd1 - fd2)
        return fourier_difference


    def compute_fourier_descriptor(self, points, num_descriptors):
        complex_points = np.empty(points.shape[0], dtype=complex)
        complex_points.real = points[:, 0]
        complex_points.imag = points[:, 1]
        descriptors = np.fft.fft(complex_points)
        descriptors = np.abs(descriptors[:num_descriptors])

        return descriptors
