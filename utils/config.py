import datetime
import os
from dataclasses import dataclass
from typing import List

import torch
import yaml
from loguru import logger


@dataclass
class Configuration:
    # 基本配置
    data_mode: str                  # 数据模式，real_scene
    num_gpus: int                   # 使用的 GPU 数量
    cuda_device_index: str          # 使用的 CUDA 设备索引
    data_dir: str                   # 数据集存储目录
    log_root_dir: str               # 日志文件的根目录
    checkpoint_root_dir: str        # 储存训练权重的根目录
    log_every_n_steps: int          # 每 n 步记录一次日志
    check_val_every_n_epoch: int    # 每 n 个 epoch 验证一次模型

    # 训练参数
    epochs: int             # 训练的总轮数
    learning_rate: float    # 学习率
    weight_decay: float     # 权重衰减，用于正则化，防止过拟合
    batch_size: int         # 每个批次的样本数量
    num_workers: int        # 数据加载时使用的工作线程数量

    # 数据处理参数
    training_dir: str           # 训练数据目录
    validation_dir: str         # 验证数据目录
    autoregressive_points: int  # 自回归模型中使用的点数
    item_number: int            
    token_nums: int
    xy_max: float
    process_dim: List[int]      # 图像处理维度，包括宽和高

    # 目标处理参数
    use_fuzzy_target: bool          # 是否使用模糊目标
    bev_encoder_in_channel: int     # BEV（鸟瞰视图）编码器的输入通道数

    # BEV 相关参数
    bev_x_bound: List[float]    # BEV 的边界
    bev_y_bound: List[float]
    bev_z_bound: List[float]
    d_bound: List[float]        # 深度边界
    final_dim: List[int]        # 最终维度
    bev_down_sample: int        # BEV 的下采样因子
    backbone: str               # 使用的主干网络

    # transformer参数
    tf_de_dim: int          # 解码器的维度
    tf_de_heads: int        # 解码器的头数
    tf_de_layers: int       # 解码器的层数
    tf_de_dropout: float    # 解码器的 dropout 比例

    # 附加参数
    append_token: int
    traj_downsample_stride: int

    # 噪声处理参数
    add_noise_to_target: bool       # 是否向目标添加噪声
    target_noise_threshold: float   # 目标噪声的阈值

    # 融合和解码参数
    fusion_method: str          # 融合方法
    decoder_method: str         # 解码方法
    query_en_dim: int           # 嵌入维度
    query_en_heads: int         # 多头注意力机制的头数
    query_en_layers: int        # 堆叠解码器层层数
    query_en_dropout: float     # dropout 比率，用于防止过拟合
    query_en_bev_length: int    # 序列长度
    target_range: float         # 嵌入维度

    # 设备和路径参数
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 指定使用的设备（CUDA 或 CPU），默认值为可用的 CUDA 设备
    resume_path: str = None     # 用于恢复训练的路径
    config_path: str = None     # 配置文件的路径
    log_dir: str = None         # 日志目录
    checkpoint_dir: str = None  # 检查点目录
    use_depth_distribution: bool = False    # 是否使用深度分布
    tf_en_motion_length: str = None


# 数据类
@dataclass
class InferenceConfiguration:
    model_ckpt_path: str    # 储存模型参数文件 .ckpt 的路径
    training_config: str    # 训练配置文件的名称或路径
    predict_mode: str       # 指定推理的模式

    trajectory_pub_frequency: int   # 轨迹发布频率
    cam_info_dir: str               # 存储相机信息的目录路径
    progress_threshold: float       # 进度阈值

    train_meta_config: Configuration = None     # 训练时的超参数配置

# 加载训练配置文件
def get_train_config_obj(config_path: str):
    exp_name = get_exp_name()
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            config_obj = Configuration(**config_yaml)
            # 设置配置路径
            config_obj.config_path = config_path
            # 设置日志目录
            config_obj.log_dir = os.path.join(config_obj.log_root_dir, exp_name)
            # 设置检查点目录
            config_obj.checkpoint_dir = os.path.join(config_obj.checkpoint_root_dir, exp_name)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    return config_obj


# 生成一个基于当前日期和时间的实验名称，输出示例：exp_2023_10_05_14_30_45
def get_exp_name():
    today = datetime.datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)
    return exp_name

# 加载推理配置文件并返回一个配置对象
def get_inference_config_obj(config_path: str):
    with open(config_path, 'r') as yaml_file:
        try:
            # 解析 YAML
            config_yaml = yaml.safe_load(yaml_file)
            # 创建配置对象
            inference_config_obj = InferenceConfiguration(**config_yaml)
        # 错误处理
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)

    # 构建训练配置路径，路径由配置文件的目录和训练配置名称组成
    training_config_path = os.path.join(os.path.dirname(config_path), "{}.yaml".format(inference_config_obj.training_config))
    # 加载训练配置
    inference_config_obj.train_meta_config = get_train_config_obj(training_config_path)
    return inference_config_obj