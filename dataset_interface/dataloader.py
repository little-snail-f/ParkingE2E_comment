import random

import numpy as np
import pytorch_lightning as pl
import torch

from dataset_interface.dataset_interface import get_parking_data
from utils.config import Configuration

from torch.utils.data import DataLoader


# 处理停车数据集的加载
class ParkingDataloaderModule(pl.LightningDataModule):
    # 构造函数
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        # 获取数据模块类 return ParkingDataModuleReal
        ParkingDataModule = get_parking_data(data_mode="real_scene")
        
        # 训练数据加载器
        self.train_loader = DataLoader(dataset=ParkingDataModule(config=self.cfg, is_train=1),
                                       batch_size=self.cfg.batch_size,      # 1
                                       shuffle=True,                        # 在每个 epoch 开始时打乱数据，以提高训练效果
                                       num_workers=self.cfg.num_workers,    # 8，工作线程数，用于并行加载数据
                                       pin_memory=True,
                                       worker_init_fn=self.seed_worker,
                                       drop_last=True)                      # 如果数据集的大小不能被批量大小整除，则丢弃最后一个不完整的批次
        self.val_loader = DataLoader(dataset=ParkingDataModule(config=self.cfg, is_train=0),
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=self.cfg.num_workers, 
                                     pin_memory=True,
                                     worker_init_fn=self.seed_worker,
                                     drop_last=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
