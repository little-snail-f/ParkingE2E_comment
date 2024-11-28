#include <memory>
#include <queue>

#include <boost/endian/conversion.hpp>
#include <boost/function/function_fwd.hpp>
#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <sensor_msgs/Image.h>

#include "fisheye_avm.hpp"
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class avm_nodelet : public nodelet::Nodelet {
 public:
 private:
  virtual void onInit() override;

  // camera结构体：存储每个摄像头的相关信息
  // 话题，参数，订阅，图像
  struct camera {
    std::string topic;
    std::string param;
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub;
    std::queue<std::pair<int, cv::Mat>> imgs;
  };

  // 回调函数，处理来自四个鱼眼摄像头的压缩图像消息
  void images_callback(sensor_msgs::CompressedImageConstPtr msg_back,
                                    sensor_msgs::CompressedImageConstPtr msg_front,
                                    sensor_msgs::CompressedImageConstPtr msg_left,
                                    sensor_msgs::CompressedImageConstPtr msg_right);

 private:
  // 四个摄像头实例，分别对应后视、前视、左视和右视摄像头
  camera back;
  camera front;
  camera left;
  camera right;

  // 输出topic
  std::string output_topic;

  // 输出图像尺寸
  int output_width;
  int output_height;
  double output_scale;

  int interpolation_mode;
  bool logging;

  FisheyeAVM avm;

  sensor_msgs::ImagePtr msg;
  cv::Mat img;

  image_transport::Publisher pub;
  int image_num = 0;

  // 图像同步策略
  // message_filters::sync_policies::ApproximateTime 
  // 是 ROS 中的一个同步策略，允许多个消息订阅者的消息在时间上近似匹配。
  // ApproximateTime 策略被应用于四个 CompressedImage 消息中
  using Policy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
      sensor_msgs::CompressedImage, sensor_msgs::CompressedImage>;
  std::shared_ptr<message_filters::Synchronizer<Policy>> p_sync;
};

// 宏定义，用于从 ROS 参数服务器获取参数
#define PARAM(name, var)                                     \
  do {                                                       \
    if (!nhp.getParam(name, var) && !nh.getParam(name, var)) {                          \
      NODELET_ERROR_STREAM("missing parameter '" #name "'"); \
      return;                                                \
    }                                                        \
  } while (0)

// ROS Nodelet 的初始化
void avm_nodelet::onInit() {
  NODELET_INFO_STREAM(__PRETTY_FUNCTION__);

  // 获取当前 Nodelet 的私有节点句柄，用于访问私有参数
  auto &nhp = getMTPrivateNodeHandle();
  // 获取当前 Nodelet 的公共节点句柄，用于访问公共参数
  auto &nh = getMTNodeHandle();

  // 使用 PARAM 宏从参数服务器获取各种参数，包括鱼眼摄像头的主题、参数文件、输出设置
  PARAM("fisheye_back_topic", back.topic);
  PARAM("fisheye_front_topic", front.topic);
  PARAM("fisheye_left_topic", left.topic);
  PARAM("fisheye_right_topic", right.topic);

  PARAM("back_param", back.param);
  PARAM("front_param", front.param);
  PARAM("left_param", left.param);
  PARAM("right_param", right.param);

  PARAM("output_topic", output_topic);
  PARAM("output_width", output_width);
  PARAM("output_height", output_height);
  PARAM("output_scale", output_scale);
  PARAM("interpolation_mode", interpolation_mode);
  PARAM("logging", logging);

  // AVM（全景图像实例）初始化
  avm.open({back.param, front.param, left.param, right.param}); // 加载参数文件
  avm.set_output_resolution({output_width, output_height}); // 设置输出图像的分辨率
  avm.set_scale(output_scale);
  avm.set_interpolation_mode(interpolation_mode);
  avm.set_logging(logging);
  avm.initialize();

  // 创建一个图像传输对象，用于发布图像消息
  image_transport::ImageTransport it(nhp);
  pub = it.advertise(output_topic, 1);

  // 为每个摄像头创建一个订阅者，接收图像消息
  back.sub.subscribe(nhp, back.topic, 1);
  front.sub.subscribe(nhp, front.topic, 1);
  left.sub.subscribe(nhp, left.topic, 1);
  right.sub.subscribe(nhp, right.topic, 1);

  // 创建一个消息同步器，使用之前定义的同步策略
  p_sync = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(10));
  // 将四个订阅者连接到同步器
  p_sync->connectInput(back.sub, front.sub, left.sub, right.sub);
  // 注册回调函数，当所有输入消息都可用时，该函数将被调用
  p_sync->registerCallback(&avm_nodelet::images_callback, this);
}

// 处理来自四个鱼眼摄像头的压缩图像消息，并生成一个合成的全景图像
void avm_nodelet::images_callback(sensor_msgs::CompressedImageConstPtr msg_back,
                                  sensor_msgs::CompressedImageConstPtr msg_front,
                                  sensor_msgs::CompressedImageConstPtr msg_left,
                                  sensor_msgs::CompressedImageConstPtr msg_right)
{
  // 图像解码，使用 cv_bridge 库将压缩的 ROS 图像消息转换为 OpenCV 格式的图像
  cv_bridge::CvImagePtr cv_ptr_back = cv_bridge::toCvCopy(msg_back, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_front = cv_bridge::toCvCopy(msg_front, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_left = cv_bridge::toCvCopy(msg_left, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_right = cv_bridge::toCvCopy(msg_right, "bgr8");

  // 从 cv_bridge 转换后的指针中提取 OpenCV 图像数据
  cv::Mat cv_back = cv_ptr_back->image;
  cv::Mat cv_front = cv_ptr_front->image;
  cv::Mat cv_left = cv_ptr_left->image;
  cv::Mat cv_right = cv_ptr_right->image;

  // 将四个摄像头的图像存储在一个 std::vector 中
  std::vector<cv::Mat> imgs = {cv_back, cv_front, cv_left, cv_right};

  // 创建一个新的 ROS 图像消息
  // 设置其头部信息（时间戳和序列号）以及图像的高度、宽度、编码格式等
  msg = boost::make_shared<sensor_msgs::Image>();
  msg->header.stamp = msg_back->header.stamp;
  msg->header.seq = msg_back->header.seq;
  msg->height = output_height;
  msg->width = output_width;
  msg->encoding = "bgr8"; // 每个像素由 8 位的蓝色、绿色和红色通道组成，OpenCV 中常用的图像格式
  msg->is_bigendian =     // 设置图像数据的字节序
      (boost::endian::order::native == boost::endian::order::big);
  msg->step = msg->width * 3; // 设置每行图像数据的字节数。对于 RGB 图像，每个像素占用 3 个字节，因此每行的字节数为 width * 3
  msg->data.resize(msg->width * msg->height * 3); // 调整 msg->data 的大小，以便存储整个图像的数据

  // 创建 OpenCV 图像
  img = cv::Mat(msg->height, msg->width, CV_8UC3, msg->data.data(), msg->step);

  // 图像处理
  avm(imgs, img);

  // 发布处理后的图像
  #ifndef USE_CUDA
  size_t dataSize = img.rows * img.cols * img.channels();
  std::memcpy(msg->data.data(), img.data, dataSize);
  #endif

  pub.publish(msg);


}

// ROS Nodelet 的一个宏，用于将 avm_nodelet 类注册为一个可插拔的 Nodelet
PLUGINLIB_EXPORT_CLASS(avm_nodelet, nodelet::Nodelet);