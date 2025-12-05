#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdlib>
#include <string>
#include <cstring>
// ROS 2 Includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
// Gazebo Includes (libgz-transport13-*, libgz-msgs10-dev)
#include <gz/transport/Node.hh>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/boolean.pb.h>
// ZeroMQ Includes (requires cppzmq installed with libzmq3-dev)
#include <zmq.hpp> 

using namespace std::chrono_literals;

// Corresponds to Python struct.unpack('iI') (int32, uint32)
#pragma pack(push, 1)
struct ClockPayload {
    int32_t sec;
    uint32_t nanosec;
};
#pragma pack(pop)

class ZMQBridge : public rclcpp::Node {
public:
    ZMQBridge() : Node("zmq_bridge_node"), context_(1), socket_(context_, zmq::socket_type::rep) {

        this->declare_parameter("step_size", 250); // How many multiples of the timestep in the world SDF (250Hz/4ms for PX4, 500Hz/2ms for ArduPilot)
        step_size_ = this->get_parameter("step_size").as_int();
        RCLCPP_INFO(this->get_logger(), "Gazebo Multi-Step Size configured to: %d", step_size_);
        
        // 1. ZMQ Setup
        socket_.bind("tcp://*:5555"); // '*' binds to all interfaces (equivalent to 0.0.0.0)
        RCLCPP_INFO(this->get_logger(), "ZMQ REP socket bound to port 5555");

        // 2. ROS 2 Setup
        publisher_ = this->create_publisher<std_msgs::msg::Float64>("/control_input", 10);
        
        subscription_ = this->create_subscription<rosgraph_msgs::msg::Clock>(
            "/clock", 10, 
            std::bind(&ZMQBridge::clock_callback, this, std::placeholders::_1));

        const char* env_val = std::getenv("WORLD");
        if (env_val == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable 'WORLD' was not set.");
        }
        std::string world_name = env_val;
        service_topic_ = "/world/" + world_name + "/control";

        // 3. Start ZMQ Thread
        running_ = true;
        zmq_thread_ = std::thread(&ZMQBridge::zmq_listener, this);
    }

    ~ZMQBridge() {
        running_ = false;
        // Clean shutdown for ZMQ
        context_.close(); 
        if (zmq_thread_.joinable()) {
            zmq_thread_.join();
        }
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr subscription_;
    
    gz::transport::Node gz_node_;
    std::string service_topic_;

    std::thread zmq_thread_;
    std::atomic<bool> running_;

    std::mutex clock_mutex_;
    std::condition_variable clock_cv_;
    bool clock_ready_ = false;
    ClockPayload current_clock_;

    int step_size_;

    void clock_callback(const rosgraph_msgs::msg::Clock::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(clock_mutex_);
        
        current_clock_.sec = msg->clock.sec;
        current_clock_.nanosec = msg->clock.nanosec;
        
        clock_ready_ = true;
        clock_cv_.notify_one(); // Signal the waiting ZMQ thread
    }

    void zmq_listener() {
        RCLCPP_INFO(this->get_logger(), "ZMQ listener thread started.");
        
        zmq::pollitem_t items[] = {
            { static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0 }
        };

        while (rclcpp::ok() && running_) {
            zmq::poll(&items[0], 1, 100); // Poll with 100ms timeout to allow checking running_ flag

            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    // 1. Receive Action
                    zmq::message_t request;
                    auto res = socket_.recv(request, zmq::recv_flags::none);
                    if (!res) continue;
                    double force = *static_cast<double*>(request.data()); // Get force (double)

                    // 2. Reset ready flag and Publish Action
                    {
                        std::lock_guard<std::mutex> lock(clock_mutex_);
                        clock_ready_ = false;
                    }
                    auto ros_msg = std_msgs::msg::Float64();
                    ros_msg.data = force;
                    publisher_->publish(ros_msg);

                    // 3. Step Gazebo
                    gz::msgs::WorldControl req;
                    req.set_multi_step(static_cast<unsigned int>(step_size_));
                    req.set_pause(true);
                    gz::msgs::Boolean rep;
                    bool result;
                    unsigned int timeout = 1000; 
                    bool executed = gz_node_.Request(service_topic_, req, timeout, rep, result);
                    if (!executed) {
                        RCLCPP_WARN(this->get_logger(), "Gazebo service call failed or timed out.");
                    }

                    // 4. Wait for ROS topic update
                    std::unique_lock<std::mutex> lock(clock_mutex_);
                    bool received = clock_cv_.wait_for(lock, 2000ms, [this]{ return clock_ready_; }); // Wait up to 2 seconds for clock_ready_ to become true

                    // 5. Send Reply
                    zmq::message_t reply(sizeof(ClockPayload));
                    if (received) {
                        // Copy struct directly into ZMQ message buffer
                        std::memcpy(reply.data(), &current_clock_, sizeof(ClockPayload));
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Update timeout!");
                        ClockPayload empty = {0, 0};
                        std::memcpy(reply.data(), &empty, sizeof(ClockPayload));
                    }
                    socket_.send(reply, zmq::send_flags::none);

                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "ZMQ Error: %s", e.what());
                }
            }
        }
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZMQBridge>());
    rclcpp::shutdown();
    return 0;
}
