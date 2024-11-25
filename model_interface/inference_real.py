from collections import OrderedDict
import numpy as np
import threading
import time

import rospy
import torch
import torchvision
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_srvs.srv import SetBool

from model_interface.model.parking_model_real import ParkingModelReal
from utils.camera_utils import CameraInfoParser, ProcessImage, get_normalized_torch_image, get_torch_intrinsics_or_extrinsics
from utils.config import InferenceConfiguration
from utils.pose_utils import PoseFlow, pose2customize_pose
from utils.ros_interface import RosInterface
from utils.traj_post_process import calculate_tangent, fitting_curve
from utils.trajectory_utils import detokenize_traj_point


# 该类主要用于处理泊车 inference 的逻辑
class ParkingInferenceModuleReal:
    # 初始化，接收推理配置和RosInterface 对象（包含启动的线程，相机、位姿处理、目标点处理、命令处理）
    def __init__(self, inference_cfg: InferenceConfiguration, ros_interface_obj: RosInterface):
        self.cfg = inference_cfg
        self.model = None
        self.device = None

        self.images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")

        self.ros_interface_obj = ros_interface_obj

        # 预训练模型 路径："./ckpt/pretrained_model.ckpt"
        self.load_model(self.cfg.model_ckpt_path)   
        
        # 序列的开始标记   1200
        self.BOS_token = self.cfg.train_meta_config.token_nums

        # 轨迹起始点信息
        self.traj_start_point_info = Pose()     
        self.traj_start_point_lock = threading.Lock()
        
        # 相机信息解析
        camera_info_obj = CameraInfoParser(task_index=-1, parser_dir=self.cfg.cam_info_dir)
        # 将解析得到的内外参分别赋值给实例属性，以便在图像处理和推理中使用   
        self.intrinsic, self.extrinsic = camera_info_obj.intrinsic, camera_info_obj.extrinsic
        
        # 序列的结束标记
        self.EOS_token = self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2

        # publisher
        self.pub = rospy.Publisher("e2e_traj_pred_topic", Path, queue_size=1)

    # 启动停车推理模块的不同工作模式（话题模式或服务模式）
    def predict(self, mode="service"):
        # 话题模式，在 ROS 中持续发布数据
        if mode == "topic":
            self.pub_topic()
        # 服务模式，等待服务请求的到来
        elif mode == "service":
            rospy.Service("/e2e_parking/srv_start", SetBool, self.pub_srv)
            rospy.spin()
        else:
            assert print("Can't support %s mode!".format(mode))

    def pub_srv(self, msg=None):
        self.get_start_pose()
        self.pub_path()
        return [True, "OK"]

    # 话题发布
    def pub_topic(self):
        # 持续检查，直到获取到 RViz 中的目标点
        while not self.ros_interface_obj.get_rviz_target():
            time.sleep(1)

        # 轨迹发布频率，1Hz
        rate = rospy.Rate(self.cfg.trajectory_pub_frequency)

        # 持续发布数据，直到 ROS 主节点关闭
        while not rospy.is_shutdown():
            self.get_start_pose()   # 获取起始位姿
            self.pub_path()         # 发布路径
            rate.sleep()

    # 生成和发布车辆轨迹
    def pub_path(self):
        # 获取和处理图像及其相关的内外参数
        images, intrinsics, extrinsics = self.get_format_images()
        # 字典
        data = {
            "image": images,            # 图像
            "extrinsics": extrinsics,   # 外参
            "intrinsics": intrinsics    # 内参
        }

        # 目标点
        target_point_info = self.ros_interface_obj.get_rviz_target()
        # 将目标点的 Z 坐标设置为当前轨迹起始点的 Z 坐标，以确保在同一平面上
        target_point_info.position.z = self.traj_start_point_info.position.z
        # 将目标点转换为自定义的位姿格式(x, y, z, yaw, pitch, roll)
        target_in_world = pose2customize_pose(target_point_info)

        # 起点（自车当前位姿势）
        ego2world = pose2customize_pose(self.traj_start_point_info)
        # 获取从世界坐标系到自车坐标系的变换矩阵
        world2ego_mat = ego2world.get_homogeneous_transformation().get_inverse_matrix()
        # 目标点在自车坐标系中的位姿
        target_in_ego = target_in_world.get_pose_in_ego(world2ego_mat)

        target_point = [[target_in_ego.x, target_in_ego.y]]

        # 将目标点和模糊目标点转换为 PyTorch 张量
        data["target_point"] = torch.from_numpy(np.array(target_point).astype(np.float32))
        data["fuzzy_target_point"] = data["target_point"]
        # 生成起始标记
        start_token = [self.BOS_token]
        # 将 start_token 转换为一个 PyTorch 张量，并将其存储在 data 字典的 gt_traj_point_token 键下
        data["gt_traj_point_token"] = torch.tensor([start_token], dtype=torch.int64).cuda()

        # 模型推理
        self.model.eval()   # 在哪？？？ 将模型切换到评估模式，可能在推理时会禁用某些训练时特有的行为，例如 dropout 和 batch normalization 
        delta_predicts = self.inference(data)

        # 轨迹后处理
        # 曲线拟合，生成平滑的轨迹
        delta_predicts = fitting_curve(delta_predicts, num_points=self.cfg.train_meta_config.autoregressive_points, item_number=self.cfg.train_meta_config.item_number)
        # 计算轨迹航向信息
        traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")

        # 创建并发布 ROS 消息
        msg = Path()
        msg.header.frame_id = "base_link"
        # 遍历预测的轨迹点和航向信息，将每个点转换为位姿并添加到消息中
        for (point_item, traj_yaw) in zip(delta_predicts, traj_yaw_path):
            # 判断轨迹点维度，进行不同的处理
            if self.cfg.train_meta_config.item_number == 2:
                x, y = point_item
            elif self.cfg.train_meta_config.item_number == 3:
                x, y, progress_bar = point_item
                if abs(progress_bar) < 1 - self.cfg.progress_threshold:
                    break
            # 添加位置信息到消息中
            msg.poses.append(self.get_posestamp_info(x, y, traj_yaw))
        # 设置时间戳
        msg.header.stamp = rospy.Time.now()
        self.pub.publish(msg)

    # 根据输入数据进行模型推理
    def inference(self, data):
        delta_predicts = []
        # 禁用梯度计算，因为在推理阶段不需要计算梯度，从而节省内存和计算资源
        with torch.no_grad():
            # 根据配置中的 decoder_method 选择相应的推理方法
            if self.cfg.train_meta_config.decoder_method == "transformer":
                delta_predicts = self.inference_transformer(data)
            elif self.cfg.train_meta_config.decoder_method == "gru":
                delta_predicts = self.inference_gru(data)
            else:
                raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        # 将预测结果从 PyTorch 张量转换为 Python 列表
        delta_predicts = delta_predicts.tolist()
        return delta_predicts

    # 使用 Transformer 进行轨迹预测
    def inference_transformer(self, data):
        # 调用模型进行预测                                              要预测的标记数量 = 一次推理中希望最终生成的轨迹点的数量 * 在自回归解码过程中每次生成的点的数量
        pred_traj_point, _, _ = self.model.predict_transformer(data, predict_token_num=self.cfg.train_meta_config.item_number*self.cfg.train_meta_config.autoregressive_points)
        # 提取出第一个样本的预测轨迹点，并去掉开始标记
        pred_traj_point_update = pred_traj_point[0][1:]
        # 移除预测轨迹中结束标记及之后的内容
        pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        # 将轨迹点从标记形式转换为实际的轨迹点坐标
        delta_predicts = detokenize_traj_point(pred_traj_point_update, self.cfg.train_meta_config.token_nums, 
                                            self.cfg.train_meta_config.item_number, 
                                            self.cfg.train_meta_config.xy_max) # 12
 
        return delta_predicts

    def inference_gru(self, data):
        delta_predicts = self.model.predict_gru(data)

        return delta_predicts

    # 移除预测轨迹中结束标记及之后的内容
    def remove_invalid_content(self, pred_traj_point_update):
        finish_index = -1
        # 查找结束标记的前一个标记的索引
        index_tensor = torch.where(pred_traj_point_update == self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2)[0]
        # 如果找到
        if len(index_tensor):
            # 查找结束标记的索引
            finish_index = torch.where(pred_traj_point_update == self.EOS_token)[0][0].item()
            # 计算有效的结束标记索引
            finish_index = finish_index - finish_index % self.cfg.train_meta_config.item_number
        if finish_index != -1:
            pred_traj_point_update = pred_traj_point_update[: finish_index]
        return pred_traj_point_update

    # 生成一个包含位置信息和方向信息的 PoseStamped 对象
    def get_posestamp_info(self, x, y, yaw):
        predict_pose = PoseStamped()
        # 将航向角转换为四元数
        pose_flow_obj = PoseFlow(att_input=[yaw, 0, 0], type="euler", deg_or_rad="deg")
        quad = pose_flow_obj.get_quad()
        # 设置位置和方向
        predict_pose.pose.position = Point(x=x, y=y, z=0.0)
        predict_pose.pose.orientation = Quaternion(x=quad.x, y=quad.y,z=quad.z, w=quad.w)
        return predict_pose

    # 获取当前的起始位姿信息
    def get_start_pose(self):
        self.traj_start_point_lock.acquire()
        tmp_start_point_info = None
        cnt = 0

        # 获取位姿
        while tmp_start_point_info == None:
            tmp_start_point_info = self.ros_interface_obj.get_pose()
            
            if cnt > 10: time.sleep(1)
            cnt += 1
        
        # 更新起始位姿（就是自车位置）
        self.traj_start_point_info = tmp_start_point_info
        self.traj_start_point_lock.release()

    # 根据标签获取对应的图像
    def get_images(self, img_tag):
        images = None
        cnt = 0
        while images == None:
            images = self.ros_interface_obj.get_images(img_tag)

            if cnt > 10: time.sleep(1)
            cnt += 1
        return images

    # 加载停车模型及其权重
    def load_model(self, parking_pth_path):
        # 设备选择，CUDA 可用，选择 GPU；否则选择 CPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 创建模型实例，将训练元配置传递给构造函数
        self.model = ParkingModelReal(self.cfg.train_meta_config)
        # 加载训练好的权重，将模型加载到第一个 CUDA 设备上（如果可用），否则会自动处理
        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        # 处理状态字典
        state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        # 加载权重
        self.model.load_state_dict(state_dict)
        # 将模型移动到之前选择的设备（CPU 或 GPU），确保模型在正确的硬件上进行推理
        self.model.to(self.device)
        # 将模型设置为评估模式
        # 这会禁用 dropout 和 batch normalization 等训练时特性，确保推理时的行为一致
        self.model.eval()

    # 获取和处理图像及其相关的内外参数
    def get_format_images(self):
        # 从配置中获取图像的目标宽度和高度
        process_width, process_height = int(self.cfg.train_meta_config.process_dim[0]), int(self.cfg.train_meta_config.process_dim[1])
        images, intrinsics, extrinsics = [], [], []

        # 遍历图像标签
        for image_tag in self.images_tag:
            # 根据标签获取原始图像，并将图像从 PyTorch 张量转换为 PIL 图像格式
            pil_image = self.torch2pillow()(self.get_images(image_tag))
            # 创建图像处理对象
            image_obj = ProcessImage(pil_image,                     # PIL 图像
                                     self.intrinsic[image_tag],     # 内参
                                     self.extrinsic[image_tag],     # 外参
                                     target_size=(process_width, process_height))  # 目标尺寸

            # 调整图像的大小并更新内参
            image_obj.resize_pil_image()
            # 将调整后的图像转换为归一化的 PyTorch 张量，并将其添加到 images 列表中
            images.append(get_normalized_torch_image(image_obj.resize_img))
            # 将更新后的内参转换为 PyTorch 张量，并添加到 intrinsics 列表中
            intrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.resize_intrinsics))
            # 将外参转换为 PyTorch 张量，并添加到 extrinsics 列表中
            extrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.extrinsics))

        # 将列表中的所有张量沿第一个维度（通常是批次维度）合并为一个张量
        # 在最前面增加一个维度
        images = torch.cat(images, dim=0).unsqueeze(0)  # (1, C, H, W) ——> (1, 4, C, H, W)
        intrinsics = torch.cat(intrinsics, dim=0).unsqueeze(0)
        extrinsics = torch.cat(extrinsics, dim=0).unsqueeze(0)

        return images, intrinsics, extrinsics

    # 将 PyTorch 张量转换为 PIL 图像格式
    # PIL 图像：可视化的图像格式，以便进行展示或保存
    def torch2pillow(self):
        return torchvision.transforms.transforms.ToPILImage()

