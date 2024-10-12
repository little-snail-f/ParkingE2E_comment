# 进行停车场预测推理
import argparse
import rospy

import utils.fix_libtiff
# 停车场推理模型
from model_interface.model_interface import get_parking_model
from utils.config import get_inference_config_obj
from utils.ros_interface import RosInterface

# 只有在直接运行该脚本时，以下代码才会执行
if __name__ == "__main__":
    # 解析命令行参数
    arg_parser = argparse.ArgumentParser()
    # 指定推理配置文件的路径
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    args = arg_parser.parse_args()

    # 初始化 ROS 节点
    rospy.init_node("e2e_traj_pred")

    # 创建 ROS 接口对象
    ros_interface_obj = RosInterface()
    threads = ros_interface_obj.receive_info()

    # 加载推理配置
    inference_cfg = get_inference_config_obj(args.inference_config_path)

    # 获取停车推理模型
    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")

    # 创建停车推理对象并进行预测
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg, ros_interface_obj=ros_interface_obj)
    parking_inference_obj.predict(mode=inference_cfg.predict_mode)

    # 等待线程结束
    for thread in threads:
        thread.join()