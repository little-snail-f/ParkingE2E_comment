import threading  # 多线程处理

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from utils.keyboard_command import CommandThread

class RosInterface:
    # 初始化
    def __init__(self):
        self.bridge = CvBridge()  # CvBridge 对象，用于图像转换

        # 变量，存储图像、位姿和目标点信息
        self.cam_front_tensor_img = None
        self.cam_left_tensor_img = None
        self.cam_right_tensor_img = None
        self.cam_rear_tensor_img = None
        self.pose_info = None
        self.target_point_info = None
        self.rviz_target = None

        # 为每个图像和信息创建锁，以确保线程安全
        self.cam_front_tensor_img_lock = threading.Lock()
        self.cam_left_tensor_img_lock = threading.Lock()
        self.cam_right_tensor_img_lock = threading.Lock()
        self.cam_rear_tensor_img_lock = threading.Lock()
        self.pose_info_lock = threading.Lock()
        self.target_point_lock = threading.Lock()
        self.rviz_target_lock = threading.Lock()

    # 图像线程函数
    # 根据相机标签订阅相应的图像话题，并调用相应的回调函数处理图像
    def image_thread_function(self, camera_label):
        camera_tag = camera_label.lower().split("_")[1]
        topic_name = "/driver/pinhole_vitual/{}".format(camera_tag)
        callback_str = "self.{}_callback_function".format(camera_tag)
        rospy.Subscriber(topic_name, Image, eval(callback_str), queue_size=1)
        rospy.spin()

    # 位姿线程函数
    # 订阅车辆位姿信息的 ROS 话题
    def pose_thread_function(self):
        rospy.Subscriber("/ego_pose", PoseStamped, self.pose_callback, queue_size=1)

    # 目标点线程函数
    # 订阅目标点信息和 RViz 目标点的 ROS 话题
    def target_point_thread_function(self):
        rospy.Subscriber("/e2e_parking/set_target_point", Bool, self.target_point_callback, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_target_callback, queue_size=1)

    # 图像回调函数
    # 将接收到的 ROS 图像消息转换为 OpenCV 格式，并将其转换为张量格式，使用锁确保线程安全
    def image_general_callback(self, msg, bri, img_lock, tag="no"):
        cv_img = bri.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img_lock.acquire()
        general_img = self.cv_img2tensor_img(cv_img)
        img_lock.release()
        return general_img

    # 位姿回调函数
    # 更新车辆的位姿信息，使用锁确保线程安全
    def pose_callback(self, msg: PoseStamped):
        self.pose_info_lock.acquire()
        self.pose_info = msg.pose
        self.pose_info_lock.release()

    # 目标点回调函数
    def target_point_callback(self, msg):
        self.target_point_info = Pose()
        self.target_point_lock.acquire()
        self.target_point_info = self.pose_info
        self.target_point_lock.release()

    # RViz 目标回调函数
    def rviz_target_callback(self, msg):
        self.rviz_target_lock.acquire()
        self.rviz_target = msg.pose
        # print(msg.pose)

        self.rviz_target_lock.release()

    # 图像转换，将 OpenCV 图像转换为 PyTorch 张量格式，进行归一化处理
    def cv_img2tensor_img(self, img_cv):
        img_cv = np.float32(np.clip(img_cv / 255, 0, 1))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img_cv, ((2, 0, 1)))).cuda()
        return img_tensor

    # 特定相机的回调函数
    def front_callback_function(self, msg):
        self.cam_front_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_front_tensor_img_lock, tag="rgb_front")

    def left_callback_function(self, msg):
        self.cam_left_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_left_tensor_img_lock, tag="rgb_left")

    def right_callback_function(self, msg):
        self.cam_right_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_right_tensor_img_lock, tag="rgb_right")

    def back_callback_function(self, msg):
        self.cam_rear_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_rear_tensor_img_lock, tag="rgb_rear")

    # 接收信息
    # 创建多个线程来处理图像、位姿和目标点信息的接收，并返回线程列表
    def receive_info(self):
        # 列表，存储所有启动的线程
        threads = []

        # 循环遍历相机标签: 对于每个相机标签（前、左、右、后），创建一个新的线程
        for camera_label in ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"]:
            image_thread = threading.Thread(target=self.image_thread_function, args=(camera_label, ))
            image_thread.start()
            threads.append(image_thread)

        # 启动位姿处理线程
        pose_thread = threading.Thread(target=self.pose_thread_function)
        pose_thread.start()
        threads.append(pose_thread)

        # 启动目标点处理线程
        target_point_thread = threading.Thread(target=self.target_point_thread_function)
        target_point_thread.start()
        threads.append(target_point_thread)

        # 启动命令处理线程
        command_thread = CommandThread([])
        command_thread.start()
        threads.append(command_thread)

        # 返回包含所有启动线程的列表
        return threads
    
    # 获取图像信息
    # 根据标签返回相应的图像张量
    def get_images(self, image_tag):
        if image_tag == "rgb_front":
            return self.cam_front_tensor_img
        elif image_tag == "rgb_rear":
            return self.cam_rear_tensor_img
        elif image_tag == "rgb_left":
            return self.cam_left_tensor_img
        elif image_tag == "rgb_right":
            return self.cam_right_tensor_img
        
    # 获取位姿信息
    def get_pose(self) -> Pose:
        return self.pose_info

    # 获取 rviz 中的目标点
    def get_rviz_target(self) -> PoseStamped:
        return self.rviz_target
