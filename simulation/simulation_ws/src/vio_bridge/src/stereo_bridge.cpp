#include <array>
#include <map>
#include <mutex>
#include <string>
#include <functional>
#include <gz/msgs/image.pb.h>
#include <gz/transport/Node.hh>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// Pair only equal acquisition stamps. Never replace a capture stamp with ROS now().
// The bounded cache drops unmatched frames; it never invents stereo synchronization.
class StereoBridge : public rclcpp::Node {
 public:
  StereoBridge() : Node("vio_stereo_bridge") {
    for (int i = 0; i < 2; ++i) {
      topics_[i] = declare_parameter<std::string>("gz_topic" + std::to_string(i), "");
      if (topics_[i].empty()) throw std::runtime_error("Both Gazebo image topics are required");
      pubs_[i] = create_publisher<sensor_msgs::msg::Image>(
          "/camera/infra" + std::to_string(i + 1) + "/image_raw", rclcpp::QoS(5).reliable());
      std::function<void(const gz::msgs::Image &)> cb =
          [this, i](const gz::msgs::Image &msg) { receive(i, msg); };
      if (!transport_.Subscribe(topics_[i], cb)) throw std::runtime_error("Gazebo subscription failed");
    }
  }
  ~StereoBridge() override {
    for (const auto &topic : topics_) transport_.Unsubscribe(topic);
  }
 private:
  void receive(int side, const gz::msgs::Image &msg) {
    if (!msg.has_header() || !msg.header().has_stamp()) return;
    const auto &stamp = msg.header().stamp();
    const int64_t t = stamp.sec() * int64_t{1000000000} + stamp.nsec();
    if (msg.pixel_format_type() != gz::msgs::RGB_INT8 || !msg.width() || !msg.height() ||
        msg.step() < msg.width() * 3 || msg.data().size() < size_t(msg.step()) * msg.height()) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Expected valid RGB_INT8 image");
      return;
    }
    sensor_msgs::msg::Image out;
    out.header.stamp.sec = stamp.sec();
    out.header.stamp.nanosec = stamp.nsec();
    out.header.frame_id = "d435_infra" + std::to_string(side + 1) + "_optical_frame";
    out.height = msg.height(); out.width = msg.width();
    out.encoding = "mono8"; out.step = out.width;
    out.data.resize(size_t(out.width) * out.height);
    cv::Mat rgb(msg.height(), msg.width(), CV_8UC3,
                const_cast<char *>(msg.data().data()), msg.step());
    cv::Mat gray(out.height, out.width, CV_8UC1, out.data.data());
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    std::lock_guard<std::mutex> lock(mutex_);
    // Each stream is ordered by Gazebo; a backwards stamp denotes a simulation reset.
    if (last_seen_[side] >= 0 && t < last_seen_[side]) {
      pending_[0].clear(); pending_[1].clear(); last_published_ = -1;
      last_seen_.fill(-1);
    }
    last_seen_[side] = t;
    if (t <= last_published_) return;
    pending_[side][t] = std::move(out);
    auto other = pending_[1 - side].find(t);
    if (other != pending_[1 - side].end()) {
      pubs_[0]->publish(pending_[0].at(t));
      pubs_[1]->publish(pending_[1].at(t));
      last_published_ = t;
      for (auto &cache : pending_) cache.erase(cache.begin(), cache.upper_bound(t));
    }
    if (pending_[side].size() > 8) {
      pending_[side].erase(pending_[side].begin());
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Dropping unmatched stereo capture");
    }
  }
  std::array<std::string, 2> topics_;
  std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, 2> pubs_;
  std::array<std::map<int64_t, sensor_msgs::msg::Image>, 2> pending_;
  std::array<int64_t, 2> last_seen_{{-1, -1}};
  int64_t last_published_ = -1;
  std::mutex mutex_;
  gz::transport::Node transport_;
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StereoBridge>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
}
