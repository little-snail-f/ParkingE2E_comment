from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar

from model_interface.trainer_real import ParkingTrainingModuleReal
from model_interface.inference_real import ParkingInferenceModuleReal


# 根据运行模式和数据模式返回相应的停车模型类
def get_parking_model(data_mode, run_mode):
    if run_mode == "train":
        if data_mode == "real_scene":
            model_class = ParkingTrainingModuleReal
        else:
            raise ValueError(f"Don't support data_mode '{data_mode}'!")
    elif run_mode == "inference":
        if data_mode == "real_scene":
            model_class = ParkingInferenceModuleReal
        else:
            raise ValueError(f"Don't support data_mode '{data_mode}'!")
    else:
        raise ValueError(f"Don't support run_mode '{run_mode}'!")
    return model_class


# 训练过程中的回调函数
def setup_callbacks(cfg):
    # Checkpoint 回调
    ckpt_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                    monitor='val_loss',                         # 监控验证集上的损失，用于判断模型性能
                                    save_top_k=3,                               # 保存性能最好的三个模型
                                    mode='min',                                 # 最小化监控指标
                                    filename='{epoch:02d}-{val_loss:.2f}',      # 保存文件的命名格式，包含当前的 epoch 和验证损失
                                    save_last=True)                             # 保存最后一个训练的模型
    
    # 进度条回调，用于在训练过程中显示进度条
    progress_bar = TQDMProgressBar()

    # 模型摘要回调，显示模型的结构和参数信息，显示的层级深度为 2
    model_summary = ModelSummary(max_depth=2)

    # 学习率监控回调，记录学习率的变化，每个 epoch 记录学习率
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return [ckpt_callback, progress_bar, model_summary, lr_monitor]