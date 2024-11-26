import os
import shutil
import sys

from loguru import logger


# 初始化，设置 CUDA 环境、初始化日志记录
def init(config_obj):
    init_cuda(config_obj)
    init_log(config_obj)

# 在训练结束时结束日志记录
def finish(config_obj):
    finish_log(config_obj)


def init_cuda(config_obj):
    # 限制程序只使用 GPU 0                             0
    os.environ['CUDA_VISIBLE_DEVICES'] = config_obj.cuda_device_index
    # CUDA 操作设置为同步模式，每个 CUDA 调用都会等待完成后再继续执行，适用于调式
    # 训练建议设置为 0，以便充分利用 GPU 的异步执行能力，提高性能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 初始化日志记录系统
def init_log(config_obj):
    logger.remove()
    logger.add(config_obj.log_dir + '/training_log_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)
    logger.info("Config File: {}", config_obj.config_path)

def save_code(cfg):
    def _ignore(path, content):
        return ["carla", "ckpt", "e2e_parking", "log"]
    project_root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    shutil.copytree(project_root_dir, os.path.join(cfg.checkpoint_dir, "code"), ignore=_ignore)
    # os.path.

# 将训练过程中生成的日志文件保存到指定目录
def finish_log(config_obj):
    # 获取当前脚本所在的目录，作为项目的根目录
    project_root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # 源日志路径
    source_log_path = os.path.join(project_root_dir, "log", os.path.basename(config_obj.checkpoint_dir))
    # 目标日志路径
    target_dir_path = os.path.join(project_root_dir, os.path.join(config_obj.checkpoint_dir, "code"), "log") 
    target_log_path = os.path.join(target_dir_path, os.path.basename(config_obj.checkpoint_dir))
    # 创建目标目录
    os.makedirs(target_dir_path, exist_ok=True)
    #  将源日志目录复制到目标日志路径
    shutil.copytree(source_log_path, target_log_path)