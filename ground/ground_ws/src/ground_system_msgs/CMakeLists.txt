cmake_minimum_required(VERSION 3.8)
project(ground_system_msgs)

# ament dependencies
find_package(ament_cmake REQUIRED)

# ROS dependecies
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(geographic_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}

  "msg/DroneObs.msg"
  "msg/SwarmObs.msg"

  DEPENDENCIES
  std_msgs
  # geometry_msgs
  # geographic_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
