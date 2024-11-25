# 可视化轨迹
# 这个节点订阅了车辆的轨迹预测和自我定位信息，并将这些信息转换到一个指定的坐标系中

import sys
import copy
import rospy
import rospkg
import threading

rp = rospkg.RosPack()
workspace_root = rp.get_path('core')
sys.path.append(workspace_root[:workspace_root.find("catkin_ws")])

# ROS 消息类型，表示几何体和路径
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nav_msgs.msg import Path
from utils.pose_utils import HomogeneousTrans, PoseFlow


class VisualizeTrajectory:
    # 初始化，设置坐标系
    def __init__(self, frame_id):
        self.frame_id = frame_id

    # 将 ROS 位姿转换为齐次变换矩阵
    def get_homo_mat_from_ros_pose(self, pose):
        return HomogeneousTrans(position_list=[pose.position.x, pose.position.y, pose.position.z], 
                                att_input=[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z], type="quad", deg_or_rad="deg")

    # 将接收到的轨迹数据转换到一个新的坐标系中，并发布转换后的轨迹
    def traj_callback(self, msg: Path):
        # 使用线程锁，确保在处理轨迹时不会同时访问 self.ego_pose_msg，以避免数据竞争
        self.ego_pose_msg_lock.acquire()
        # 持续检查 self.ego_pose_msg 是否有值，直到有定位消息可用
        while not self.ego_pose_msg:
            continue
        cnt = 0
        while self.ego_pose_msg:
            cnt += 1
            if cnt > 10:
                break

        # 将车辆的定位位姿转换为齐次变换矩阵，用于后续的坐标转换
        ego2map_matrix = self.get_homo_mat_from_ros_pose(self.ego_pose_msg.pose).get_matrix()
        
        # 深拷贝路径消息
        pub_path = copy.deepcopy(msg)
        pub_path.header.frame_id = self.frame_id
        # 遍历路径中的每个点
        # 1.将其位姿转换为齐次变换矩阵
        # 2.然后使用矩阵乘法将当前点的坐标从自车定位坐标系转换到地图坐标系
        # 3.最后，将转换后的位姿（坐标和方向）更新到路径点中
        for item in pub_path.poses:
            cur_mat = self.get_homo_mat_from_ros_pose(item.pose).get_matrix()
            ret_mat = ego2map_matrix @ cur_mat
            x, y, z = ret_mat[:3,-1].tolist() # 提取平移部分
            qw, qx, qy, qz = PoseFlow(att_input=ret_mat[:3,:3], type="rot_mat").get_quad() # 提取旋转部分并转换为四元数
            item.pose.position = Point(x=x, y=y, z=z)
            item.pose.orientation = Quaternion(x=qx, y=qy,z=qz, w=qw)

        self.predict_traj_pub.publish(pub_path)

        self.ego_pose_msg_lock.release()

    # 处理自车定位消息，并发布当前的定位位姿
    def localization_callback(self, msg: PoseStamped):
        self.ego_pose_msg = msg
        cur_pose_stamp = self.get_stamped_ego_pose(msg.pose)
        self.ego_pose_pub.publish(cur_pose_stamp)

    # 创建带有 frame_id 的位姿消息（没有时间戳吗）
    def get_stamped_ego_pose(self, pose):
        # 创建 PoseStamped 消息实例
        cur_pose_stamp = PoseStamped()
        cur_pose_stamp.header.frame_id = self.frame_id
        cur_pose_stamp.pose = pose
        return cur_pose_stamp

    # 主函数
    def main(self):
        # 初始化节点
        rospy.init_node("e2e_traj_show")
        
        #     话题订阅      topic                 消息类型   处理函数          消息队列大小，实时
        rospy.Subscriber("/e2e_traj_pred_topic", Path, self.traj_callback, queue_size=1)
        rospy.Subscriber("/ego_pose", PoseStamped, self.localization_callback, queue_size=1)
       
        #                               话题发布  topic                        消息类型  消息队列大小，实时
        self.predict_traj_pub = rospy.Publisher("e2e_traj_pred_in_map_topic", Path, queue_size=1)
        self.ego_pose_pub = rospy.Publisher("ego", PoseStamped, queue_size=1) 

        # 初始化线程锁和定位消息
        self.ego_pose_msg_lock = threading.Lock()
        self.ego_pose_msg = None
        
        # 进入循环，等待并处理回调函数，函数会一直运行，直到节点被关闭
        rospy.spin()

# Python 入口，通常用于执行一个类的实例化和方法调用
if __name__ == "__main__":
    # 实例化 VisualizeTrajectory 类
    obj = VisualizeTrajectory(frame_id="iekf_map")
    # 调用 main 方法
    obj.main()