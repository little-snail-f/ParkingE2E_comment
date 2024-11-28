import argparse    # 处理命令行参数和选项

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_interface.dataloader import ParkingDataloaderModule
from model_interface.model_interface import get_parking_model, setup_callbacks
from utils.config import get_train_config_obj
from utils.decorator_train import finish, init


# 装饰器函数
# 用于在调用训练函数之前和之后执行一些初始化和清理操作
def decorator_function(train_function):
    # 包装函数
    def wrapper_function(*args, **kwargs):
        # 设置 CUDA 环境、初始化日志记录
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        # 结束日志记录，将训练过程中生成的日志文件保存到指定目录
        finish(*args, **kwargs)
    return wrapper_function


# 使用 PyTorch Lightning 进行训练
@decorator_function
def train(config_obj):
    # 使用 PyTorch Lightning 创建 Trainer 实例
    # Trainer 是 PyTorch Lightning 中的一个核心类，用于管理训练过程，封装了训练循环、验证、测试等功能
    parking_trainer = Trainer(callbacks=setup_callbacks(config_obj),                # 设置回调函数，包括Checkpoint 回调、进度条回调、模型摘要回调和学习率监控回调
                              logger=TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False),
                              accelerator='gpu',                                    # 使用 GPU 进行训练
                              strategy='ddp' if config_obj.num_gpus > 1 else None,  # 如果使用多个 GPU，则使用分布式数据并行（DDP）策略
                              devices=config_obj.num_gpus,                          # GPU 数量
                              max_epochs=config_obj.epochs,                         # 最大训练轮数
                              log_every_n_steps=config_obj.log_every_n_steps,
                              check_val_every_n_epoch=config_obj.check_val_every_n_epoch,   # 每 5 个 epoch 验证一次模型
                              profiler='simple')                                    # 使用简单的性能分析器
    
    # 获取模型模块 -- ParkingTrainingModuleReal                          "real_scene"
    ParkingTrainingModelModule = get_parking_model(data_mode=config_obj.data_mode, run_mode="train")

    # 模型实例，包括超参数、损失函数、模型结构
    model = ParkingTrainingModelModule(config_obj)

    # 数据加载器实例，初始化训练和验证数据加载器为 None
    data = ParkingDataloaderModule(config_obj)

    # 训练
    parking_trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)


def main():
    # 设置随机种子，以确保训练过程的可重复性
    seed_everything(16)

    # 命令行参数解析，读取训练配置文件
    arg_parser = argparse.ArgumentParser() # 创建解析器
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str) # 添加参数
    args = arg_parser.parse_args()  # 解析命令行参数
    config_path = args.config       # 配置文件路径
    # 加载配置对象
    config_obj = get_train_config_obj(config_path)

    # 开始训练
    train(config_obj)  # 先跳转到 wrapper_function 进行包装


if __name__ == '__main__':
    main()