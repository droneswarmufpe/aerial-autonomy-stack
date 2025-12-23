#include "ground_system.hpp"

GroundSystem::GroundSystem() : Node("ground_system"), keep_running_(true)
{
    // Declare Parameters
    this->declare_parameter("num_drones", 1);
    this->declare_parameter("ip", "0.0.0.0");
    this->declare_parameter("base_port", 14540);
    this->declare_parameter("rate", 10.0);

    // Get Parameters
    num_drones_ = this->get_parameter("num_drones").as_int();
    ip_ = this->get_parameter("ip").as_string();
    base_port_ = this->get_parameter("base_port").as_int();
    publish_rate_ = this->get_parameter("rate").as_double();

    // Random Seed
    rng_.seed(std::random_device()());

    // Publisher
    publisher_ = this->create_publisher<ground_system_msgs::msg::SwarmObs>("/tracks", 10);

    // Timer
    auto timer_period = std::chrono::duration<double>(1.0 / publish_rate_);
    timer_ = this->create_wall_timer(timer_period, std::bind(&GroundSystem::publish_swarm_obs, this));

    // Start Listener Threads
    for (int i = 0; i < num_drones_; ++i) {
        int drone_id = i + 1;
        int port = base_port_ + i;
        listener_threads_.emplace_back(&GroundSystem::mavlink_listener, this, drone_id, port);
        RCLCPP_INFO(this->get_logger(), "Listening for drone %d on port %d", drone_id, port);
    }
}

GroundSystem::~GroundSystem()
{
    keep_running_ = false;
    for (auto &t : listener_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void GroundSystem::mavlink_listener(int drone_id, int port)
{
    // Setup UDP Socket
    int sockfd;
    struct sockaddr_in servaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        RCLCPP_ERROR(this->get_logger(), "Socket creation failed for drone %d", drone_id);
        return;
    }

    // Set timeout for recvfrom so the thread can exit cleanly
    struct timeval read_timeout;
    read_timeout.tv_sec = 1;
    read_timeout.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof(read_timeout));

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(ip_.c_str());
    servaddr.sin_port = htons(port);

    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        RCLCPP_ERROR(this->get_logger(), "Bind failed for drone %d on port %d", drone_id, port);
        close(sockfd);
        return;
    }

    uint8_t buffer[2048];
    mavlink_message_t msg;
    mavlink_status_t status;

    while (keep_running_ && rclcpp::ok()) {
        ssize_t len = recvfrom(sockfd, (char *)buffer, 2048, 0, NULL, NULL);
        // Prevention of CPU hogging is handled by recvfrom blocking/timeout
        if (len > 0) {
            // Parse bytes
            for (ssize_t i = 0; i < len; ++i) {
                if (mavlink_parse_char(MAVLINK_COMM_0, buffer[i], &msg, &status)) {
                    
                    if (msg.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT) { // Handle GLOBAL_POSITION_INT message
                        mavlink_global_position_int_t pos;
                        mavlink_msg_global_position_int_decode(&msg, &pos);
                        DroneData obs;
                        obs.lat = pos.lat / 1e7;
                        obs.lon = pos.lon / 1e7;
                        obs.alt = pos.alt / 1000.0; // mm to m
                        obs.vx = pos.vx / 100.0;    // cm/s to m/s
                        obs.vy = pos.vy / 100.0;
                        obs.vz = pos.vz / 100.0;
                        {
                            std::lock_guard<std::mutex> lock(data_mutex_);
                            drone_obs_[drone_id] = obs;
                        }
                    }
                }
            }
        } else if (len < 0) { // Handle timeout or error
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue; // Just a timeout, continue loop to check keep_running_
            } else {
                RCLCPP_WARN(this->get_logger(), "Recv failed for drone %d: %s", drone_id, strerror(errno));
            }
        }
    }

    close(sockfd);
}

void GroundSystem::publish_swarm_obs()
{
    const double POS_STD_DEV_DEG = 1e-5;
    const double ALT_STD_DEV_M = 0.5;
    const double VEL_STD_DEV_MS = 0.1;

    ground_system_msgs::msg::SwarmObs swarm_msg;
    swarm_msg.header.stamp = this->now();

    // Copy data to minimize lock duration
    std::map<int, DroneData> current_obs;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_obs = drone_obs_;
    }

    for (const auto &pair : current_obs) {
        int id = pair.first;
        DroneData track = pair.second;

        ground_system_msgs::msg::DroneObs drone_msg;
        drone_msg.id = id;
        drone_msg.label = (id == 2) ? 48 : 0; // HARDCODED: example where drone 2 is given label 48, the talking dead

        drone_msg.latitude_deg = add_noise(track.lat, POS_STD_DEV_DEG);
        drone_msg.longitude_deg = add_noise(track.lon, POS_STD_DEV_DEG);
        drone_msg.altitude_m = add_noise(track.alt, ALT_STD_DEV_M);
        drone_msg.velocity_n_m_s = add_noise(track.vx, VEL_STD_DEV_MS);
        drone_msg.velocity_e_m_s = add_noise(track.vy, VEL_STD_DEV_MS);
        drone_msg.velocity_d_m_s = add_noise(track.vz, VEL_STD_DEV_MS);

        swarm_msg.tracks.push_back(drone_msg);
    }

    if (!swarm_msg.tracks.empty()) {
        publisher_->publish(swarm_msg);
    }
}

double GroundSystem::add_noise(double value, double std_dev)
{
    std::normal_distribution<double> dist(0.0, std_dev);
    return value + dist(rng_);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GroundSystem>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
