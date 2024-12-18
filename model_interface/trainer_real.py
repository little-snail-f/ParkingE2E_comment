import pytorch_lightning as pl
import torch

from loss.traj_point_loss import TokenTrajPointLoss, TrajPointLoss
from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import Configuration
from utils.metrics import CustomizedMetric


class ParkingTrainingModuleReal(pl.LightningModule):
    # 构造函数
    def __init__(self, cfg: Configuration):
        # 调用父类构造函数
        super(ParkingTrainingModuleReal, self).__init__()
        # 保存超参数
        self.save_hyperparameters()

        self.cfg = cfg

        # 设置损失函数
        self.traj_point_loss_func = self.get_loss_function()
        
        # 创建泊车模型的实例
        self.parking_model = ParkingModelReal(self.cfg)


    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)

        train_loss = self.traj_point_loss_func(pred_traj_point, batch)        

        loss_dict.update({"train_loss": train_loss})

        self.log_dict(loss_dict)

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)

        val_loss = self.traj_point_loss_func(pred_traj_point, batch)

        val_loss_dict.update({"val_loss": val_loss})

        customized_metric = CustomizedMetric(self.cfg, pred_traj_point, batch)
        val_loss_dict.update(customized_metric.calculate_distance(pred_traj_point, batch))

        self.log_dict(val_loss_dict)

        return val_loss

    # 配置优化器
    def configure_optimizers(self):
        # 创建优化器(Adam)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.learning_rate,   # 学习率 0.0001
                                     weight_decay=self.cfg.weight_decay)    # 权重衰减（L2 正则化），用于防止过拟合
        # 创建学习率调度器(余弦退火学习率调度器) ———— 逐渐降低学习率                            一个周期的最大迭代次数？？
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # 根据配置选择适当的损失函数
    def get_loss_function(self):
        traj_point_loss_func = None
        if self.cfg.decoder_method == "transformer":
            traj_point_loss_func = TokenTrajPointLoss(self.cfg)
        elif self.cfg.decoder_method == "gru":
            traj_point_loss_func = TrajPointLoss(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        return traj_point_loss_func