import os
from PIL import Image
from typing import List

import numpy as np
import torch.utils.data
import tqdm

from utils.camera_utils import CameraInfoParser, ProcessImage, get_normalized_torch_image, get_torch_intrinsics_or_extrinsics
from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, tokenize_traj_point


class ParkingDataModuleReal(torch.utils.data.Dataset):
    # 构造函数
    def __init__(self, config: Configuration, is_train):
        super(ParkingDataModuleReal, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums                                # 1200                               
        self.EOS_token = self.cfg.token_nums + self.cfg.append_token - 2    # 1201
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1    # 1202

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train                                            # 1
        self.images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")

        self.intrinsic = {}
        self.extrinsic = {}
        self.images = {}
        for image_tag in self.images_tag:    # {'rgb_front': [], 'rgb_left': [], 'rgb_right': [], 'rgb_rear': []}
            self.images[image_tag] = []

        self.task_index_list = []

        self.fuzzy_target_point = []
        self.traj_point = []
        self.traj_point_token = []
        self.target_point = []
        self.create_gt_data()               # 地面真值数据

    def __len__(self):
        return len(self.images["rgb_front"])

    def __getitem__(self, index):
        images, intrinsics, extrinsics = self.process_camera(index)

        data = {}
        keys = ['image', 'extrinsics', 'intrinsics', 'target_point', 'gt_traj_point', 'gt_traj_point_token', 'fuzzy_target_point']
        for key in keys: 
            data[key] = []
        data['image'] = images
        data['intrinsics'] = intrinsics
        data['extrinsics'] = extrinsics
        data["gt_traj_point"] = torch.from_numpy(np.array(self.traj_point[index]))
        data['gt_traj_point_token'] = torch.from_numpy(np.array(self.traj_point_token[index]))
        data['target_point'] = torch.from_numpy(self.target_point[index])
        data["fuzzy_target_point"] = torch.from_numpy(self.fuzzy_target_point[index])

        return data

    # 生成地面真值（ground truth）数据，包括轨迹点、目标点和图像路径等
    def create_gt_data(self):
        # 获取所有任务的列表（训练数据的路径）
        all_tasks = self.get_all_tasks()

        # 任务迭代
        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):  # 使用 tqdm 库为任务迭代添加进度条，便于监控处理进度
            # 解析与当前任务相关的相机信息和轨迹信息
            image_info_obj = CameraInfoParser(task_index, task_path)   # 训练数据索引，训练数据路径，当前训练数据的相机内外参
            traje_info_obj = TrajectoryInfoParser(task_index, task_path) # 轨迹点数量、轨迹点、可能的目标点

            # 相机的内参和外参
            self.intrinsic[task_index] = image_info_obj.intrinsic
            self.extrinsic[task_index] = image_info_obj.extrinsic

            # 轨迹点迭代
            for ego_index in range(0, traje_info_obj.total_frames):  # ego iteration
                # 获取轨迹点和变换矩阵
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix() # [4, 4]
                # create predict point 预测点
                predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat)
                # create parking goal 停车目标
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat)
                # create image_path 图像路径
                image_path = self.create_image_path_gt(task_path, ego_index)

                # 将生成的预测点、停车目标和模糊停车目标添加到相应的列表中
                self.traj_point.append(predict_point_gt)
                self.traj_point_token.append(predict_point_token_gt)
                self.target_point.append(parking_goal)
                self.fuzzy_target_point.append(fuzzy_parking_goal)
                # 将生成的图像路径添加到相应的图像列表中
                for image_tag in self.images_tag:
                    self.images[image_tag].append(image_path[image_tag])
                self.task_index_list.append(task_index)

        # 
        self.format_transform()

    def process_camera(self, index):
        process_width, process_height = int(self.cfg.process_dim[0]), int(self.cfg.process_dim[1])
        images, intrinsics, extrinsics = [], [], []
        for image_tag in self.images_tag:
            image_path_list = self.images[image_tag]
            image_obj = ProcessImage(self.load_pil_image(image_path_list[index]), 
                                     self.intrinsic[self.task_index_list[index]][image_tag], 
                                     self.extrinsic[self.task_index_list[index]][image_tag], 
                                     target_size=(process_width, process_height))
            image_obj.resize_pil_image()
            images.append(get_normalized_torch_image(image_obj.resize_img))
            intrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.resize_intrinsics))
            extrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.extrinsics))
        images = torch.cat(images, dim=0)
        intrinsics = torch.cat(intrinsics, dim=0)
        extrinsics = torch.cat(extrinsics, dim=0)

        return images, intrinsics, extrinsics

    # 基于给定的轨迹信息和变换矩阵，生成预测点和相应的标记（token）
    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array) -> List[int]:
        predict_point, predict_point_token = [], []
        # 遍历 30 个的自回归预测点(步幅为 3)
        for predict_index in range(self.cfg.autoregressive_points):  # predict iteration 30
            # 当前预测点的索引？？？ 3，6，9，12，15，...，90
            predict_stride_index = self.get_clip_stride_index(predict_index = predict_index, 
                                                                start_index=ego_index,  # 当前真实轨迹点索引
                                                                max_index=traje_info_obj.total_frames - 1, 
                                                                stride=self.cfg.traj_downsample_stride) # 3
            # 获取当前预测索引对应的轨迹点并转换到自车坐标系
            predict_pose_in_world = traje_info_obj.get_trajectory_point(predict_stride_index)
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
            # 获取当前预测点的进度
            progress = traje_info_obj.get_progress(predict_stride_index)
            # 存储预测点
            predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            # 生成标记（token） 维度:[x, y, progress]
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y, 
                                                progress, self.cfg.token_nums, self.cfg.xy_max)
            # 维度:[x, y]
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            # 如果达到轨迹的最后一帧或预测点的最后一个索引，则提前结束循环
            if predict_stride_index == traje_info_obj.total_frames - 1 or predict_index == self.cfg.autoregressive_points - 1:
                break

        # python 列表推导式，将嵌套列表 predict_point 展平为一个一维列表 predict_point_gt
        # 首先遍历 predict_point 中的每个子列表 sublist，然后遍历每个子列表中的每个元素 item
        predict_point_gt = [item for sublist in predict_point for item in sublist]
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)
        assert append_pad_num >= 0
        predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]

        predict_point_token_gt = [item for sublist in predict_point_token for item in sublist]
        predict_point_token_gt.insert(0, self.BOS_token)
        predict_point_token_gt.append(self.EOS_token)
        predict_point_token_gt.append(self.PAD_token)
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number + self.cfg.append_token - len(predict_point_token_gt)
        assert append_pad_num >= 0
        predict_point_token_gt = predict_point_token_gt + append_pad_num * [self.PAD_token]
        return predict_point_token_gt, predict_point_gt

    def create_parking_goal_gt(self, traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array):
        candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
        candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
        fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]

        target_pose_in_world = traje_info_obj.get_precise_target_pose()
        target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
        parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y]

        return fuzzy_parking_goal, parking_goal


    def create_image_path_gt(self, task_path, ego_index):
        filename = f"{str(ego_index).zfill(4)}.png"
        image_path = {}
        for image_tag in self.images_tag:
            image_path[image_tag] = os.path.join(task_path, image_tag, filename)
        return image_path

    # 获取训练或验证数据集中的所有任务路径
    def get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir) # './e2e_dataset/train'
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir) # './e2e_dataset/val'
        # 根据配置和当前训练状态构建训练和验证数据的目录路径
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
        # 遍历指定数据目录中的所有场景中的所有任务
        for scene_item in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_item)
            for task_item in os.listdir(scene_path):
                task_path = os.path.join(scene_path, task_item)
                # 将每个任务的完整路径添加到 all_tasks 列表中
                all_tasks.append(task_path) # './e2e_dataset/train/demo_bag/1708690461_right'
        return all_tasks

    # 将存储的地面真值数据转换为 NumPy 数组
    def format_transform(self):
        # 将轨迹点转换为 NumPy 数组，并将数据类型设置为 float32
        self.traj_point = np.array(self.traj_point).astype(np.float32)
        self.traj_point_token = np.array(self.traj_point_token).astype(np.int64)
        self.target_point = np.array(self.target_point).astype(np.float32)
        self.fuzzy_target_point = np.array(self.fuzzy_target_point).astype(np.float32)
        for image_tag in self.images_tag:
            self.images[image_tag] = np.array(self.images[image_tag]).astype(np.string_)
        self.task_index_list = np.array(self.task_index_list).astype(np.int64)

    # 计算新的索引：start_index + stride * (1 + predict_index)，确保计算结果在 0 和 max_index 之间
    def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
        return int(np.clip(start_index + stride * (1 + predict_index), 0, max_index))

    def load_pil_image(self, image_path):
        return Image.open(image_path).convert("RGB")