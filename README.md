# ParkingE2E

## ParkingE2E: Camera-based End-to-end Parking Network, from Images to Planning
Autonomous parking is a crucial task in the intelligent driving field.
Traditional parking algorithms are usually implemented using rule-based schemes.
However, these methods are less effective in complex parking scenarios due to the intricate design of the algorithms.
In contrast, neural-network-based methods tend to be more intuitive and versatile than the rule-based methods.
By collecting a large number of expert parking trajectory data and emulating human strategy via learning-based methods, the parking task can be effectively addressed.
We employ imitation learning to perform end-to-end planning from RGB images to path planning by imitating human driving trajectories.
The proposed end-to-end approach utilizes a target query encoder to fuse images and target features, and a transformer-based decoder to autoregressively predict future waypoints.

**视频:**

<img src="resource/video_show.gif" height="250">

补充视频资料可在以下网址获得： [\[Link\]](https://youtu.be/urOEHJH1TBQ).

**Related Papers:**

- Changze Li, Ziheng Ji, Zhe Chen, Tong Qin and Ming Yang. "ParkingE2E: Camera-based End-to-end Parking Network, from Images to Planning." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024. [\[Link\]](https://arxiv.org/pdf/2408.02061)

- Yunfan Yang, Denglon Chen, Tong Qin, Xiangru Mu, Chunjing Xu, Ming Yang. "E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator." 2024 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2024. [\[Link\]](https://ieeexplore.ieee.org/abstract/document/10588551)


## 1. 先决条件
Ubuntu 20.04, CUDA, ROS Noetic and OpenCV 4.


## 2. Setup
Clone the code:
```Shell
git clone https://github.com/ChauncyLeee/e2e_parking_imitation.git
cd e2e_parking_imitation/
```

Install virtual environment:
```Shell
conda env create -f environment.yaml
```

Setup interface:
```shell
conda activate ParkingE2E
PARKINGE2E_PYTHON_PATH=`which python`
cd catkin_ws
catkin_make -DPYTHON_EXECUTABLE=${PARKINGE2E_PYTHON_PATH}
source devel/setup.bash
```


## 3. 运行

#### 下载预训练模型和测试数据:
首先，您需要下载[预训练模型](https://drive.google.com/file/d/1rZ4cmgXOUFgJDLFdnvAI6voU9ZkhsmYV/view?usp=drive_link) and [测试数据](https://drive.google.com/file/d/11kA-srYa6S30OqyCdyg3jGNZxBMsUHYC/view?usp=drive_link). 然后，您需要修改`./config/inference_real.yaml`中的推理配置 `model_ckpt_path`。

#### 运行驱动程序:
```Shell
roslaunch core driven_core.launch
```

首次执行该命令时会出现进度条（用于计算畸变图），待四个（鱼眼相机）进度条完成后，即可进行后续操作。

#### 使用 E2E 算法开始推理:
```shell
conda activate ParkingE2E
python ros_inference.py
```
第一次执行该命令时，EfficientNet将下载预先训练的模型。

#### 运行测试demo:
```shell
unzip demo_scene.zip
cd demo_scene
# scene_index = 1, 2, 3, 4, 5, 6, 7. For example: sh ./demo.sh 1
sh ./demo.sh ${scene_index}
```

In rviz, you can also select the parking trarget using `2D nav goal` on the rviz pannel.

<img src="resource/demo.gif" height="250">

## 4. 训练
我们提供[demo rosbag](https://drive.google.com/file/d/1jIG1iRMeW9XXdWP7eEJKnZP1gC0xvG7o/view?usp=drive_link)来创建一个迷你数据集并训练一个模型。
#### 生成数据集
首先，您需要创建一个数据集。
```
python toolkit/dataset_generation.py --bag_file_path ${DEMO_BAG_PATH} --output_folder_path ./e2e_dataset
```
如果你使用自己的rosbag，请确认 `./catkin_ws/src/core/config/params.yaml` 中的rosbag主题，并修改相机配置。

#### 训练你的模型
```Shell
python train.py --config ./config/training_real.yaml
```
您可以在 `./config/training_real.yaml` 中修改训练配置。


## 5. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.