import torch
from torch import nn

from model_interface.model.bev_encoder import BevEncoder, BevQuery
from model_interface.model.gru_trajectory_decoder import GRUTrajectoryDecoder
from model_interface.model.lss_bev_model import LssBevModel
from model_interface.model.trajectory_decoder import TrajectoryDecoder
from utils.config import Configuration


class ParkingModelReal(nn.Module):
    # 构造函数
    def __init__(self, cfg: Configuration):
        # 调用父类的构造函数
        super().__init__()

        self.cfg = cfg

        # 模型组件初始化
        # Camera Encoder
        self.lss_bev_model = LssBevModel(self.cfg) # 结合了 EfficientNet 作为主干网络和 DeepLabHead 结构来提取多尺度特征。。通过上采样和特征拼接，该模型能够有效地处理图像分割任务。
        self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel) # 基于 ResNet 的设计，具体使用了基本块（BasicBlock）来构建网络  

        # Target Encoder
        self.target_res_encoder = BevEncoder(in_channel=1)

        # BEV Query 
        # BEV 查询模块，包含 TransformerDecoder 组件
        self.bev_query = BevQuery(self.cfg)

        # Trajectory Decoder
        self.trajectory_decoder = self.get_trajectory_decoder()

    def forward(self, data):
        # Encoder
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="train")

        # Decoder
        pred_traj_point = self.trajectory_decoder(bev_feature, data['gt_traj_point_token'].cuda())

        return pred_traj_point, pred_depth, bev_target

    # 通过 Transformer 进行轨迹预测
    # 输入：输入数据的字典[图像特征、目标点、内外参]，要预测的标记数量
    def predict_transformer(self, data, predict_token_num):
        # Encoder
        # 调用编码器，输出融合了图像特征和目标特征的 BEV 特征，预测深度，目标特征
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="predict")

        # Auto Regressive Decoder
        # 自回归解码器
        autoregressive_point = data['gt_traj_point_token'].cuda() # 在推理过程中，通常将开始标记（BOS）视为目标轨迹点 During inference, we regard BOS as gt_traj_point_token.
        # 生成轨迹点
        for _ in range(predict_token_num):
            # 生成下一个轨迹点
            pred_traj_point = self.trajectory_decoder.predict(bev_feature, autoregressive_point)
            # 将新生成的轨迹点 pred_traj_point 连接到 autoregressive_point 的末尾
            autoregressive_point = torch.cat([autoregressive_point, pred_traj_point], dim=1)

        return autoregressive_point, pred_depth, bev_target

    def predict_gru(self, data):
        # Encoder
        bev_feature, _, _ = self.encoder(data, mode="predict")

        # Decoder
        autoregressive_point = self.trajectory_decoder(bev_feature).squeeze()
        return autoregressive_point

    # 处理输入数据，提取相机图像和目标点的特征，并进行特征融合
    def encoder(self, data, mode):
        # Camera Encoder -- 相机编码
        # 将输入的图像、内参和外参数据移动到指定的设备
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        # 调用模型 forward 方法生成 BEV 特征和深度信息
        bev_camera, pred_depth = self.lss_bev_model(images, intrinsics, extrinsics)
        # 对 BEV 特征进行编码，不对特征进行展平处理
        bev_camera_encoder = self.image_res_encoder(bev_camera, flatten=False)
    
        # Target Encoder -- 目标点编码
        # 选择目标点，并将目标点数据转移到指定的设备
        target_point = data['fuzzy_target_point'] if self.cfg.use_fuzzy_target else data['target_point']
        target_point = target_point.to(self.cfg.device, non_blocking=True)
        # 将目标点在 BEV 图中表示并编码
        bev_target = self.get_target_bev(target_point, mode=mode)
        # 对目标点的 BEV 表示进行编码，flatten=False 表示保留空间结构信息
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False)
        
        # Feature Fusion -- 特征融合
        # 将编码后的目标特征和相机特征进行融合，生成最终的 BEV 特征
        bev_feature = self.get_feature_fusion(bev_target_encoder, bev_camera_encoder)

        # 对特定维度进行扁平化处理
        # [batch_size, channel, h, w] -> [batch_size, channel, h * w]
        bev_feature = torch.flatten(bev_feature, 2)

        # 返回融合了图像特征和目标特征的 BEV 特征，预测深度，目标特征
        return bev_feature, pred_depth, bev_target

    # 将目标点转换为 BEV 格式的目标张量
    # 将 BEV 图终点附近的区域置为 1
    def get_target_bev(self, target_point, mode):
        # 计算 BEV 图的高度和宽度（ = 长度/每个像素对应的实际距离）
        h, w = int((self.cfg.bev_y_bound[1] - self.cfg.bev_y_bound[0]) / self.cfg.bev_y_bound[2]), int((self.cfg.bev_x_bound[1] - self.cfg.bev_x_bound[0]) / self.cfg.bev_x_bound[2])
        # 批次大小
        b = self.cfg.batch_size if mode == "train" else 1

        # Get target point
        # 创建形状为 (b, 1, h, w) 的全零张量
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)
        # 将目标点的坐标转换为像素坐标（像素坐标 = BEV 坐标原点 + 坐标/每个像素在该方向上对应的实际距离）
        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        # 添加噪声
        # 如果配置中启用了噪声添加，并且当前模式是训练模式，则生成随机噪声并将其添加到目标点坐标中
        if self.cfg.add_noise_to_target and mode == "train":
            noise_threshold = int(self.cfg.target_noise_threshold / self.cfg.bev_x_bound[2])
            noise = (torch.rand_like(target_point, dtype=torch.float) * noise_threshold * 2 - noise_threshold).int()
            target_point += noise

        # 在 BEV 视图中设置目标点
        for batch in range(b):
            bev_target_batch = bev_target[batch][0] # (h, w)
            # 当前批次的目标点像素坐标
            target_point_batch = target_point[batch]
            # 影响范围（像素值）
            range_minmax = int(self.cfg.target_range / self.cfg.bev_x_bound[2])
            # 使用切片操作标记目标区域，将选定区域的值设置为 1.0
            # x范围：[target_point_batch[0] - range_minmax, target_point_batch[0] + range_minmax]
            # x范围：[target_point_batch[1] - range_minmax, target_point_batch[1] + range_minmax]
            bev_target_batch[target_point_batch[0] - range_minmax: target_point_batch[0] + range_minmax + 1,
                             target_point_batch[1] - range_minmax: target_point_batch[1] + range_minmax + 1] = 1.0
        return bev_target
    

    # 根据配置的融合方法将目标特征和相机特征进行融合
    def get_feature_fusion(self, bev_target_encoder, bev_camera_encoder):
        # "query" 融合
        if self.cfg.fusion_method == "query":
            bev_feature = self.bev_query(bev_target_encoder, bev_camera_encoder)
        # 加法融合
        elif self.cfg.fusion_method == "plus":
            bev_feature = bev_target_encoder + bev_camera_encoder
        # 拼接融合
        elif self.cfg.fusion_method == "concat":
            concat_feature = torch.concatenate([bev_target_encoder, bev_camera_encoder], dim=1)
            conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False).cuda()
            bev_feature = conv(concat_feature)
        else:
            raise ValueError(f"Don't support fusion_method '{self.cfg.fusion_method}'!")
        
        return bev_feature
    
    # 根据配置选择合适的轨迹解码器
    def get_trajectory_decoder(self):
        if self.cfg.decoder_method == "transformer":
            trajectory_decoder = TrajectoryDecoder(self.cfg)
        elif self.cfg.decoder_method == "gru":
            trajectory_decoder = GRUTrajectoryDecoder(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        
        return trajectory_decoder