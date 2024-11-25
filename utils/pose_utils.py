import numpy as np
import pyquaternion
from geometry_msgs.msg import Pose


class PoseFlow:
    """
    Right hand pose flow
    ROS: qx, qy,qz, qw;  pyquaternion: qw, qx, qy, qz
    """
    def __init__(self, att_input, type, deg_or_rad=None) -> None:
        """
        |--------type-------|------------att_input format----------------|
        |        quad       |               PYTHON List                  |
        |        euler      | PYTHON List, the order is yaw, pitch, roll |
        |        rot_mat    |  two-dimensional list or numpy array       |

        """
        self.quad = None
        self.deg_or_rad = deg_or_rad
        if type == "quad":
            self.quad = pyquaternion.Quaternion(att_input)
        # 欧拉角输入
        elif type == "euler":
            assert self.deg_or_rad in ["deg", "rad"]
            self.quad = self._get_quad_from_euler(*att_input)
        elif type == "rot_mat":
            self.quad = pyquaternion.Quaternion(matrix=att_input)
        else:
            assert print("Can't support!")
    
    def get_quad(self):
        """
        return: local to world quaternion
        """
        return self.quad

    def get_euler(self):
        """
        Premise 1: In pyquaternion, all rotation matrices are local to world.
        Premise 2: In pyquaternion, .yaw_pitch_roll decomposes the rotation 
                   matrix into R = R_x(roll) R_y(pitch) R_z(yaw).
        Premise 3: In single-axis rotation, negating the attitude angle 
                   is equivalent to taking the inverse of the rotation matrix. 
                   The negation of the attitude angle and the inverse of the 
                   rotation matrix are always equivalent in rotation.
        Explanation: If you want to use .yaw_pitch_roll, premise 2 specifies 
                     that the rotation order can only be from world to local, 
                     while premise 1 states that the resulting matrix is local to world.
                     Using the explanation provided in premise 3, 
                     all attitude angles must be negated to conform to the definition 
                     of attitude angles.
        """
        assert self.deg_or_rad in ["deg", "rad"]
        yaw, pitch, roll = self.quad.inverse.yaw_pitch_roll
        yaw, pitch, roll = -yaw, -pitch, -roll
        if self.deg_or_rad == "deg":
            yaw, pitch, roll = self._deg_vs_rad([yaw, pitch, roll], "rad2deg")
        return yaw, pitch, roll

    def get_rotation_matrix(self):
        """
        return: local to world rotation matrix
        """
        return self.quad.rotation_matrix

    def _deg_vs_rad(self, euler, direction):
        yaw, pitch, roll = euler
        if direction == "rad2deg":
            yaw, pitch, roll = np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
        elif direction == "deg2rad":
            yaw, pitch, roll = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
        else:
            assert print("Angle type error!")
        return yaw, pitch, roll
    
    # 将欧拉角（航向、俯仰和滚转）转换为四元数
    def _get_quad_from_euler(self, yaw, pitch, roll):
        # 如果输入角度以度为单位，将度数转换为弧度
        if self.deg_or_rad == "deg":
            yaw, pitch, roll = self._deg_vs_rad([yaw, pitch, roll], direction="deg2rad")
        # 计算四元数
        quad = pyquaternion.Quaternion(axis=[0, 0, 1], angle=yaw) * pyquaternion.Quaternion(axis=[0, 1, 0], angle=pitch) * pyquaternion.Quaternion(axis=[1, 0, 0], angle=roll)
        return quad

class Rotation:
    def __init__(self, yaw, pitch, roll) -> None:
        self.yaw, self.pitch, self.roll = yaw, pitch, roll

class HomogeneousTrans(PoseFlow):
    def __init__(self, position_list, att_input, type=None, deg_or_rad=None) -> None:
        super(HomogeneousTrans, self).__init__(att_input, type, deg_or_rad)
        self.rotation_matrix = self.get_rotation_matrix()
        self.position_list = position_list
        self.rotation = Rotation(*self.get_euler())

    # 生成一个齐次变换矩阵
    def get_matrix(self):
        homo_matrix = np.eye(4) # 创建单位矩阵
        homo_matrix[:3, -1] = np.array(self.position_list)  # 设置平移部分
        homo_matrix[:3, :3] = self.rotation_matrix          # 设置旋转部分
        return homo_matrix
    
    # 计算并返回齐次变换矩阵的逆矩阵
    def inverse(self):
        return np.linalg.inv(self.get_matrix()) # 计算逆矩阵
    
    # 计算齐次变换矩阵的逆矩阵
    def get_inverse_matrix(self):
        return self.inverse()
    

class CustomizePose:
    def __init__(self, x, y, z, roll, yaw, pitch):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch

    # 将世界坐标系中的位姿转换为自车坐标系中的位姿
    # world2ego_mat：起点处的世界坐标系到自车坐标系的变换矩阵
    def get_pose_in_ego(self, world2ego_mat: np.array):
        # 获取当前位姿的齐次变换矩阵，表示从自车坐标系到世界坐标系的变换
        pose2world_mat = self.get_homogeneous_transformation().get_matrix()
        homogeneous_matrix = world2ego_mat @ pose2world_mat
        # 提取旋转矩阵
        att_input = homogeneous_matrix[:3, :3]
        # 获取欧拉角
        pose_flow = PoseFlow(att_input=att_input, type="rot_mat", deg_or_rad="deg")
        yaw, pitch, roll = pose_flow.get_euler()
        # 提取平移部分
        position_list = homogeneous_matrix[:3, -1].tolist()
        x, y, z = position_list[0], position_list[1], position_list[2]
        return CustomizePose(x=x, y=y, z=z, roll=roll, yaw=yaw, pitch=pitch)

    # 生成齐次变换对象
    def get_homogeneous_transformation(self):
        return HomogeneousTrans(position_list=[self.x, self.y, self.z],
                                att_input=[self.yaw, self.pitch, self.roll],
                                type="euler", deg_or_rad="deg")
    

# 将 ROS 中的 Pose 信息转换为自定义的位姿格式 CustomizePose
def pose2customize_pose(pose_info: Pose) -> CustomizePose:
    # 创建对象，用于处理位姿的齐次变换
    target_homo_flow = HomogeneousTrans(
                    position_list=[pose_info.position.x, pose_info.position.y,  pose_info.position.z], 
                    att_input=[pose_info.orientation.w, pose_info.orientation.x, pose_info.orientation.y, 
                                pose_info.orientation.z], type="quad", deg_or_rad="deg")
    # 获取欧拉角
    yaw, pitch, roll = target_homo_flow.get_euler()
    # 提取位置坐标
    x, y, z = target_homo_flow.position_list
    # 创建 CustomizePose 对象
    customize_pose = CustomizePose(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
    return customize_pose