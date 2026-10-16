# Intel Realsense D435 Gazebo Harmonic

This demo uses an old D435 model originally created for Gazebo Classic. Instead of using a plugin that publishes the data to ROS directly the SDF was adapted to use the new camera and depth_camera plugins.

To get this to work download the realsense_d435.stl from https://github.com/nilseuropa/realsense_ros_gazebo/blob/master/meshes/realsense_d435.stl and create the following folder structure:
```
d435
- meshes
  - realsense_d435.stl
- model.config
- model.sdf
```