from torch import nn

from utils.config import Configuration


class TokenTrajPointLoss(nn.Module):
    # 构造函数
    def __init__(self, cfg: Configuration):
        super(TokenTrajPointLoss, self).__init__()
        self.cfg = cfg
        # 填充标记的索引，在计算损失时会被忽略  1200 + 3 - 1
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1
        # 初始化交叉熵损失函数，并指定忽略的索引
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.PAD_token)

    # 前向传播
    def forward(self, pred, data):
        pred = pred[:, :-1,:]
        pred_traj_point = pred.reshape(-1, pred.shape[-1])
        gt_traj_point_token = data['gt_traj_point_token'][:, 1:-1].reshape(-1).cuda()

        traj_point_loss = self.ce_loss(pred_traj_point, gt_traj_point_token)
        return traj_point_loss


class TrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TrajPointLoss, self).__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        gt = data['gt_traj_point'].view(-1, self.cfg.autoregressive_points, 2)
        traj_point_loss = self.mse_loss(pred, gt)
        return traj_point_loss