#include <rclcpp/rclcpp.hpp>
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include <mutex>
#include <cmath>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mavros_msgs/msg/vfr_hud.hpp>
#include <state_sharing/msg/shared_state.hpp>

class StateSharingNode : public rclcpp::Node
{
public:
    StateSharingNode() : Node("state_sharing_node")
    {
        this->declare_parameter<std::string>("autopilot", "ardupilot");
        this->declare_parameter<int>("drone_id", 1);
        autopilot = this->get_parameter("autopilot").as_string();
        drone_id_ = this->get_parameter("drone_id").as_int();

        if (this->get_parameter("use_sim_time").as_bool()) {
            RCLCPP_INFO(this->get_logger(), "Simulation time is enabled.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Simulation time is disabled.");
        }

        publisher_ = this->create_publisher<state_sharing::msg::SharedState>("/state_sharing_drone_" + std::to_string(drone_id_), 10);

        callback_group_timer_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant); // Timed callbacks in parallel
        timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::seconds(1), std::bind(&StateSharingNode::publish_timer_callback, this), callback_group_timer_);

        callback_group_subscriber_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant); // Listen to subscribers in parallel
        auto subscriber_options = rclcpp::SubscriptionOptions();
        subscriber_options.callback_group = callback_group_subscriber_;
        rclcpp::QoS qos_profile_sub(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
        qos_profile_sub.keep_last(10);  // History: KEEP_LAST with depth 10
        qos_profile_sub.reliability(rclcpp::ReliabilityPolicy::BestEffort);

        if (autopilot == "ardupilot")
        {
            subscription_navsat_apm_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
                "/mavros/global_position/global", 
                qos_profile_sub, std::bind(&StateSharingNode::ardupilot_navsat_callback, this, std::placeholders::_1), subscriber_options);

            subscription_vel_apm_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
                "/mavros/local_position/velocity_local",
                qos_profile_sub, std::bind(&StateSharingNode::ardupilot_vel_callback, this, std::placeholders::_1), subscriber_options);

            subscription_hud_apm_ = this->create_subscription<mavros_msgs::msg::VfrHud>(
                "/mavros/vfr_hud",
                qos_profile_sub, std::bind(&StateSharingNode::ardupilot_hud_callback, this, std::placeholders::_1), subscriber_options);
            
        }
        RCLCPP_INFO(this->get_logger(), "state_sharing_node initialized");
    }

private:
    int drone_id_;
    std::string autopilot;

    state_sharing::msg::SharedState latest_state_;
    std::mutex data_mutex_;

    rclcpp::CallbackGroup::SharedPtr callback_group_timer_;
    rclcpp::CallbackGroup::SharedPtr callback_group_subscriber_;

    rclcpp::Publisher<state_sharing::msg::SharedState>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subscription_navsat_apm_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr subscription_vel_apm_;
    rclcpp::Subscription<mavros_msgs::msg::VfrHud>::SharedPtr subscription_hud_apm_;
    rclcpp::TimerBase::SharedPtr timer_;


    void ardupilot_navsat_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_state_.latitude_deg = msg->latitude;
        latest_state_.longitude_deg = msg->longitude;
        latest_state_.altitude_m = msg->altitude; // This is ellipsoid altitude
    }

    void ardupilot_vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_state_.vx = msg->twist.linear.x;
        latest_state_.vy = msg->twist.linear.y;
        latest_state_.vz = msg->twist.linear.z;
    }

    void ardupilot_hud_callback(const mavros_msgs::msg::VfrHud::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_state_.heading_deg = msg->heading; // In degrees
    }

    void publish_timer_callback()
    {
        state_sharing::msg::SharedState message;
        message.header.stamp = this->get_clock()->now();
        message.drone_id = drone_id_;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            message.latitude_deg = latest_state_.latitude_deg;
            message.longitude_deg = latest_state_.longitude_deg;
            message.altitude_m = latest_state_.altitude_m;
            message.vx = latest_state_.vx;
            message.vy = latest_state_.vy;
            message.vz = latest_state_.vz;
            message.heading_deg = latest_state_.heading_deg;
        }
        publisher_->publish(message);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor; // Or set num_threads with executor(rclcpp::ExecutorOptions(), 8);
    auto node = std::make_shared<StateSharingNode>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
