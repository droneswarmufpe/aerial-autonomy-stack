cmake_minimum_required(VERSION 3.8)
project(state_sharing)

# ament dependencies
find_package(ament_cmake REQUIRED)

# ROS dependecies
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}

  "msg/SharedState.msg"

  DEPENDENCIES
  std_msgs
  geometry_msgs
  # geographic_msgs
)

add_executable(state_sharing_node src/state_sharing_node.cpp)

ament_target_dependencies(state_sharing_node rclcpp px4_msgs sensor_msgs geometry_msgs mavros_msgs)

# Link the message library
rosidl_target_interfaces(state_sharing_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS
  state_sharing_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
