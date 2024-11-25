###################################################
################# 进行泊车推理(testing) #################
###################################################

import argparse # 命令行参数解析库
import rospy

import utils.fix_libtiff
# 停车场推理(testing)模型
from model_interface.model_interface import get_parking_model
from utils.config import get_inference_config_obj
from utils.ros_interface import RosInterface

# 主程序入口
if __name__ == "__main__":      # 只有在直接运行该脚本时，以下代码才会执行
    # 解析命令行参数
    arg_parser = argparse.ArgumentParser()
    # 指定推理配置(testing)文件的路径
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    # 检查命令行中提供的参数，并将其与之前定义的参数进行匹配，然后返回一个包含所有解析参数的对象
    args = arg_parser.parse_args()

    # 初始化 ROS 节点
    rospy.init_node("e2e_traj_pred")

    # 创建 RosInterface 对象
    ros_interface_obj = RosInterface()
    # 所有已启动的线程，相机、位姿处理、目标点处理、命令处理
    threads = ros_interface_obj.receive_info()

    # 加载推理配置，包括了训练的超参数
    inference_cfg = get_inference_config_obj(args.inference_config_path)

    # 获取泊车推理模型，模型将使用训练好的权重进行 testing，而不是进行训练
    # 返回 ParkingInferenceModuleReal                                                          real_scene
    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")

    # 创建泊车 inference 对象并进行预测
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg, ros_interface_obj=ros_interface_obj)
    parking_inference_obj.predict(mode=inference_cfg.predict_mode) # mode = "topic"

    # 等待所有线程结束（里面的程序并行运行？）
    for thread in threads:
        thread.join()  # 阻塞调用它的线程(主线程)，直到被调用的线程完成执行