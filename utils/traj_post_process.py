import numpy as np
from scipy.special import comb


# 生成贝塞尔曲线             30
def bezier_curve(points, num_points, item_number):
    # 贝塞尔曲线的阶数，n次贝塞尔曲线
    n = len(points) - 1
    # 生成一个包含 num_points 个均匀分布在 [0, 1] 区间的数值的数组
    t = np.linspace(0, 1, num_points)
    # 形状为 (num_points, item_number) 的数组，存储生成的曲线点
    curve = np.zeros((num_points, item_number))

    # 遍历每个生成的曲线点
    for i in range(num_points):
        # 遍历每个控制点
        for j in range(n + 1):
            curve[i] += comb(n, j) * (1 - t[i])**(n - j) * t[i]**j * points[j]
    return curve

# 根据给定的原始数据生成平滑曲线     30          2
def fitting_curve(raw_data, num_points, item_number):
    control_points = np.array(raw_data)
    # 使用贝塞尔曲线算法生成平滑的曲线
    smoothed_curve = bezier_curve(control_points, num_points, item_number)
    return smoothed_curve.tolist()

# 计算给定点集的切线方向
def calculate_tangent(points, mode):
    num_points = len(points)

    if num_points < 2:
        return [0.]

    tangent = np.zeros((num_points, 2))
    for i in range(num_points):
        # 三点法
        if mode == "three_point":
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            elif i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            else:
                tangent[i] = -(points[i+1] - points[i-1])
        # 五点法
        elif mode == "five_point":
            # 对于第一个和最后一个点，使用相邻点计算切线
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            elif i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            # 对于第二个和倒数第二个点，使用前一个点计算切线
            elif (i == 1) or (i == num_points - 2):
                tangent[i] = -(points[i] - points[i-1])
            # 对于其他点，使用五点差分法计算切线
            else:
                tangent[i] = -((points[i - 2] - 8 * points[i - 1] + 8*points[i + 1] - points[i + 2]) / 12)
        # 三点后向法
        elif mode == "three_point_back":
            if i == 0:
                tangent[i] = -(points[i+1] - points[i])
            else:
                tangent[i] = -(points[i] - points[i-1])
        # 三点前向法
        elif mode == "three_point_front":
            if i == num_points - 1:
                tangent[i] = -(points[i] - points[i-1])
            else:
                tangent[i] = -(points[i+1] - points[i])
        else:
            assert print("Error mode!")

    traj_heading_list = np.rad2deg(np.arctan2(tangent[:, 1], tangent[:, 0])).tolist()
    
    if np.any(np.isnan(traj_heading_list)):
        print("NaN")
    return traj_heading_list

