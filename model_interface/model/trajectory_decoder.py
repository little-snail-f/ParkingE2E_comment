import torch
from torch import nn
from timm.models.layers import trunc_normal_

from utils.config import Configuration


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        # 嵌入层，用于将输入的标记（tokens）转换为稠密的向量表示
        #                             词汇表中token的数量(1200)+附加的特殊标记的数量(3)，嵌入向量的维度(256)
        #                                       例如开始标记(BOS)、结束标记(EOS)、填充标记(PAD)等 
        self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points

        # 位置嵌入(随机生成，训练更新)          批次大小，位置嵌入的长度: 2 * 30 + 2(开始标记和结束标记)，嵌入向量维度   将生成的随机数乘以 0.02，缩小初始化值的范围，避免梯度爆炸或消失
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.item_number*item_cnt + 2, self.cfg.tf_de_dim) * .02)

        # 创建单个解码层                            输入向量的维度(256)            多头数(4)
        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        # 创建完整的 Transformer 解码器                       层数(4)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)

        # 线性层，将解码器的输出映射到目标标记空间         线性层的输出特征的维度，表示模型可以预测的标记数量(1200 + 3)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums + self.cfg.append_token)

        self.init_weights()

    # 初始化权重
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    # 创建掩码 
    def create_mask(self, tgt):
        # 下三角矩阵
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        # 创建填充掩码
        # 标记目标序列中填充标记的位置，填充标记通常用于处理变长的序列，以确保模型不会在这些位置进行计算
        tgt_padding_mask = (tgt == self.PAD_token)

        return tgt_mask, tgt_padding_mask

    # 解码操作
    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        # 将图像特征和输入轨迹嵌入表示的第一列和第二列交换位置，以便与 Transformer 解码器的输入格式相匹配
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        # 调用 Transformer 解码器
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,   # 之前的轨迹嵌入
                                        memory=encoder_out,     # BEV 特征
                                        tgt_mask=tgt_mask,      # 目标掩码
                                        tgt_key_padding_mask=tgt_padding_mask)  # 填充掩码
        # 转置输出的轨迹点
        pred_traj_points = pred_traj_points.transpose(0, 1)
        return pred_traj_points

    def forward(self, encoder_out, tgt):
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)

        pred_traj_points = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)
        return pred_traj_points
    
    # 预测轨迹点
    def predict(self, encoder_out, tgt):
        # 标记的数量
        length = tgt.size(1)
        # 需要填充的数量（轨迹点数量 + 2 - 标记数量，2 代表开始标记和结束标记）
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + 2 - length
        
        offset = 1
        # 填充目标序列
        if padding_num > 0:
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to('cuda')
            tgt = torch.cat([tgt, padding], dim=1)

        # 生成目标掩码和填充掩码
        # 掩码用于在解码过程中控制模型的注意力机制，确保模型不会看到填充的和未来的标记
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        # 使用嵌入层将目标序列转换为嵌入向量
        tgt_embedding = self.embedding(tgt)
        # 将位置编码添加到目标嵌入中
        tgt_embedding = tgt_embedding + self.pos_embed

        # 解码器调用
        pred_traj_points = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        # 将解码器的输出通过线性层映射到标记空间
        # 选择最后一个时间步的输出
        pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]

        # softmax，选择概率最大的结果
        pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
        pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
        return pred_traj_points