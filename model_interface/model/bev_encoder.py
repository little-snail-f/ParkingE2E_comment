import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18

from utils.config import Configuration


# 该类处理输入的特征图并提取特征
# 使用 ResNet-18 作为主干网络
class BevEncoder(nn.Module):
    # 构造函数
    def __init__(self, in_channel):
        super().__init__()

        # 使用 resnet18 函数初始化一个 ResNet-18 主干网络
        # 不加载预训练权重，将残差块的权重初始化为零
        trunk = resnet18(weights=None, zero_init_residual=True)

        # 卷积层                输入通道     输出64   卷积核
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 批归一化层
        self.bn1 = trunk.bn1
        # ReLU 激活函数
        self.relu = trunk.relu
        # 最大池化层
        self.max_pool = trunk.maxpool

        # 残差层
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

    def forward(self, x, flatten=True):
        # 使用双线性插值将输入图像调整为 256x256 的大小
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # 输入处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if flatten:
            x = torch.flatten(x, 2)
        return x


class BevQuery(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        # Transformer 解码器层
        # 单个解码器层
        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads, batch_first=True, dropout=self.cfg.query_en_dropout)
        # 堆叠 num_layers 个解码器层  # 4
        self.tf_query = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.query_en_layers)

        # 位置嵌入，为输入序列添加位置信息 可学习参数 [256, 256]
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.query_en_bev_length, self.cfg.query_en_dim) * .02)

        # 初始化权重
        self.init_weights()

    # 初始化权重
    def init_weights(self):
        # 遍历模型参数及其对应的张量
        # 参数的名称，参数的张量
        for name, p in self.named_parameters():
            # 跳过位置嵌入初始化
            if 'pos_embed' in name:
                continue
            # 参数维度大于 1（即是权重矩阵），使用 Xavier 均匀分布进行初始化
            # Xavier 初始化旨在保持前向传播和反向传播时的信号方差相对稳定
            # 适用于使用 sigmoid 或 tanh 激活函数的网络
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 将 pos_embed 初始化为截断正态分布(标准差设置为 0.02)
        # 截断正态分布在生成的值超出某个范围时会被重新生成
        # 通常用于确保嵌入值不会过大或过小，从而有助于模型的稳定性
        trunc_normal_(self.pos_embed, std=.02)

    # 将目标特征和图像特征通过 Transformer 解码器进行融合，生成最终的 BEV 特征
    def forward(self, tgt_feature, img_feature):
        # 检查两个特征形状是否相同
        assert tgt_feature.shape == img_feature.shape
        # 批次大小，通道数，高度，宽度
        batch_size, channel, h, w = tgt_feature.shape

        # 调整目标特征和图像特征的形状
        # [batch_size, channel, h, w] -> [batch_size, channel, h * w]
        tgt_feature = tgt_feature.view(batch_size, channel, -1)
        img_feature = img_feature.view(batch_size, channel, -1)

        # 维度变换，以符合 Transformer 的输入要求
        # [batch_size, channel, h * w] -> [batch_size, h * w, channel]
        tgt_feature = tgt_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
        img_feature = img_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        # 将位置嵌入添加到目标特征和图像特征中
        tgt_feature = tgt_feature + self.pos_embed
        img_feature = tgt_feature + self.pos_embed

        # 调用 transformer 解码器
        # 将目标特征作为输入，图像特征作为记忆（memory），生成融合后的 BEV 特征
        bev_feature = self.tf_query(tgt_feature, memory=img_feature)

        # 维度恢复
        # [batch_size, h * w, channel] -> [batch_size, channel, h * w]
        bev_feature.permute(0, 2, 1)
        # [batch_size, channel, h * w] -> [batch_size, channel, h, w]
        bev_feature = bev_feature.view(batch_size, channel, h, w)
        return bev_feature


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         return x * y


# class BEVTfEncoder(nn.Module):
#     def __init__(self, cfg: Configuration, input_dim):
#         super().__init__()
#         self.cfg = cfg

#         tf_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=self.cfg.tf_en_heads)
#         self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=self.cfg.tf_en_layers)

#         self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.tf_en_bev_length, input_dim) * .02)
#         self.pos_drop = nn.Dropout(self.cfg.tf_en_dropout)

#         self.init_weights()

#     def init_weights(self):
#         for name, p in self.named_parameters():
#             if 'pos_embed' in name:
#                 continue
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         trunc_normal_(self.pos_embed, std=.02)

#     def forward(self, bev_feature, mode):
#         bev_feature = bev_feature
#         if mode == "train":
#             bev_feature = self.pos_drop(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         bev_feature = self.tf_encoder(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         return bev_feature
