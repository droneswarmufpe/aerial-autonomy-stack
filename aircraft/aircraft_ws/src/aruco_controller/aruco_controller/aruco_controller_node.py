import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cv2
import numpy as np
import os
import platform
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
try:
    from autopilot_interface_msgs.action import Land
except ImportError:
    Land = None
try:
    from mavros_msgs.srv import CommandTOL
except ImportError:
    CommandTOL = None


class ControllerState(Enum):
    INITIALIZING = 0
    WAITING_FOR_MARKER = 1
    CENTERING = 2
    DESCENDING = 3
    LANDING = 4
    COMPLETED = 5
    ERROR = -1


@dataclass
class ArucoMarkerData:
    detected: bool
    marker_id: int
    corners: Optional[np.ndarray]
    center: Optional[Tuple[float, float]]
    area: float
    timestamp: float


class ArucoCenteredDescentController(Node):
    def __init__(self):
        super().__init__('aruco_controller_node')
        
        # Declare and get parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('camera_frame_id', 'camera_frame')
        self.declare_parameter('camera_stream_port', 5600)
        self.declare_parameter('command_topic', '/mavros/setpoint_velocity/cmd_vel_unstamped')
        self.declare_parameter('aruco_marker_id', 0)
        self.declare_parameter('aruco_marker_size', 0.05)
        self.declare_parameter('kp_x', 0.3)
        self.declare_parameter('kp_y', 0.3)
        self.declare_parameter('descent_velocity', -0.5)
        self.declare_parameter('centering_tolerance_px', 20)
        self.declare_parameter('min_alignment_confidence', 0.8)
        self.declare_parameter('max_marker_area_px', 10000)
        self.declare_parameter('min_marker_area_px', 100)
        self.declare_parameter('marker_lost_timeout_sec', 2.0)
        self.declare_parameter('max_horizontal_velocity', 1.0)
        self.declare_parameter('max_descent_velocity', 1.0)
        self.declare_parameter('initial_alignment_timeout_sec', 10.0)
        self.declare_parameter('descent_timeout_sec', 30.0)
        self.declare_parameter('publish_debug_images', False)
        
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.camera_stream_port = int(self.get_parameter('camera_stream_port').value)
        self.command_topic = self.get_parameter('command_topic').value
        self.hitl = os.getenv('HITL', 'false').lower() == 'true'
        self.architecture = platform.machine()
        self.target_marker_id = self.get_parameter('aruco_marker_id').value
        self.marker_size = self.get_parameter('aruco_marker_size').value
        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.descent_velocity = self.get_parameter('descent_velocity').value
        self.centering_tolerance = self.get_parameter('centering_tolerance_px').value
        self.min_confidence = self.get_parameter('min_alignment_confidence').value
        self.max_marker_area = self.get_parameter('max_marker_area_px').value
        self.min_marker_area = self.get_parameter('min_marker_area_px').value
        self.marker_lost_timeout = self.get_parameter('marker_lost_timeout_sec').value
        self.max_horizontal_vel = self.get_parameter('max_horizontal_velocity').value
        self.max_descent_vel = self.get_parameter('max_descent_velocity').value
        self.centering_timeout = self.get_parameter('initial_alignment_timeout_sec').value
        self.descent_timeout = self.get_parameter('descent_timeout_sec').value
        self.publish_debug = self.get_parameter('publish_debug_images').value
        
        # Initialize ArUco detector
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            if hasattr(cv2.aruco, 'ArucoDetector')
            else None
        )
        
        # Image bridge
        self.bridge = CvBridge()
        
        # State management
        self.state = ControllerState.INITIALIZING
        self.data_lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.marker_data = ArucoMarkerData(
            detected=False, marker_id=-1, corners=None, center=None, area=0, timestamp=0
        )
        self.last_marker_seen_time = time.time()
        self.state_entry_time = time.time()
        
        # Control outputs
        self.velocity_cmd = Twist()
        
        # Camera intrinsics (default, may be overridden)
        self.image_width = 320
        self.image_height = 240
        self.image_center_x = self.image_width / 2.0
        self.image_center_y = self.image_height / 2.0
        
        # QoS profile for subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, self.camera_topic, self.camera_callback,
            qos_profile
        )
        self.camera_capture = None
        self.camera_stream_running = True
        
        # Publishers
        self.velocity_pub = self.create_publisher(Twist, self.command_topic, 10)
        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(Image, 'aruco_debug', 10)
        
        # Action clients
        self.land_client = ActionClient(self, Land, 'land_action') if Land is not None else None
        if self.land_client is None:
            self.get_logger().warn('autopilot_interface_msgs not available; using MAVROS land service fallback.')
        self.mavros_land_client = (
            self.create_client(CommandTOL, '/mavros/cmd/land') if CommandTOL is not None else None
        )
        if self.mavros_land_client is None:
            self.get_logger().warn('mavros_msgs not available; MAVROS land service fallback disabled.')
        
        # Main control loop timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop_callback)
        
        self.get_logger().info(f"ArUco Controller initialized. Target marker ID: {self.target_marker_id}")
        self.get_logger().info(f"Camera topic: {self.camera_topic}")
        self.get_logger().info(f"Camera stream port: {self.camera_stream_port}")
        self.get_logger().info(f"Velocity command topic: {self.command_topic}")
        self.get_logger().info(f"Marker size: {self.marker_size}m")

        self.camera_stream_thread = threading.Thread(target=self._camera_stream_loop, daemon=True)
        self.camera_stream_thread.start()

    def destroy_node(self):
        self.camera_stream_running = False
        if self.camera_stream_thread.is_alive():
            self.camera_stream_thread.join(timeout=1.0)
        if self.camera_capture is not None:
            self.camera_capture.release()
        super().destroy_node()

    def _open_camera_stream(self):
        if self.architecture == 'x86_64':
            gst_pipeline_string = (
                f"udpsrc port={self.camera_stream_port} ! "
                "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                "rtph264depay ! "
                "avdec_h264 ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )
        elif self.architecture == 'aarch64' and self.hitl:
            gst_pipeline_string = (
                f"udpsrc port={self.camera_stream_port} ! "
                "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                "rtph264depay ! "
                "h264parse ! "
                "nvv4l2decoder ! "
                "nvvidconv ! "
                "video/x-raw, format=I420 ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! "
                "appsink drop=true max-buffers=1"
            )
        else:
            return None

        cap = cv2.VideoCapture(gst_pipeline_string, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            self.get_logger().warn('Failed to open GStreamer camera stream. ROS image topic will be the only input.')
            return None
        self.get_logger().info('Opened GStreamer camera stream.')
        return cap

    def _camera_stream_loop(self):
        self.camera_capture = self._open_camera_stream()
        if self.camera_capture is None:
            return
        while rclpy.ok() and self.camera_stream_running:
            ret, frame = self.camera_capture.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.data_lock:
                self.current_frame = frame.copy()
                self.image_height, self.image_width = frame.shape[:2]
                self.image_center_x = self.image_width / 2.0
                self.image_center_y = self.image_height / 2.0
            self._detect_aruco_markers(frame)

    def camera_callback(self, msg: Image):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.data_lock:
                self.current_frame = cv_image.copy()
                self.image_height, self.image_width = cv_image.shape[:2]
                self.image_center_x = self.image_width / 2.0
                self.image_center_y = self.image_height / 2.0
            
            # Detect ArUco markers
            self._detect_aruco_markers(cv_image)
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {e}")

    def _detect_aruco_markers(self, frame: np.ndarray):
        try:
            # Detect markers
            if self.aruco_detector is not None:
                corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, self.aruco_dict, parameters=self.aruco_params
                )
            
            with self.data_lock:
                if ids is not None and self.target_marker_id in ids:
                    # Find the target marker
                    idx = np.where(ids == self.target_marker_id)[0][0]
                    marker_corners = corners[idx]
                    
                    # Compute marker center and area
                    corner_pts = marker_corners[0].astype(int)
                    center_x = np.mean(corner_pts[:, 0])
                    center_y = np.mean(corner_pts[:, 1])
                    
                    # Compute area (approximate as bounding box)
                    x_coords = corner_pts[:, 0]
                    y_coords = corner_pts[:, 1]
                    area = float((max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)))
                    
                    if area >= self.min_marker_area:
                        self.marker_data = ArucoMarkerData(
                            detected=True,
                            marker_id=self.target_marker_id,
                            corners=marker_corners[0],
                            center=(center_x, center_y),
                            area=area,
                            timestamp=time.time()
                        )
                        self.last_marker_seen_time = time.time()
                    else:
                        self.marker_data.detected = False
                else:
                    self.marker_data.detected = False
                
                # Publish debug image if requested
                if self.publish_debug and self.marker_data.detected:
                    debug_frame = self._draw_debug_info(frame.copy())
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
                    debug_msg.header.stamp = self.get_clock().now().to_msg()
                    debug_msg.header.frame_id = self.camera_frame_id
                    self.debug_image_pub.publish(debug_msg)
                    
        except Exception as e:
            self.get_logger().error(f"Error detecting ArUco markers: {e}")

    def _draw_debug_info(self, frame: np.ndarray) -> np.ndarray:
        # Draw crosshair at center
        center_x, center_y = int(self.image_center_x), int(self.image_center_y)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        if self.marker_data.detected and self.marker_data.corners is not None:
            # Draw marker corners
            corners = self.marker_data.corners.astype(int)
            for i in range(len(corners)):
                pt1 = corners[i]
                pt2 = corners[(i + 1) % len(corners)]
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
            
            # Draw marker center
            marker_cx, marker_cy = int(self.marker_data.center[0]), int(self.marker_data.center[1])
            cv2.circle(frame, (marker_cx, marker_cy), 5, (255, 0, 0), -1)
        
        # Add state text
        cv2.putText(frame, f"State: {self.state.name}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Marker detected: {self.marker_data.detected}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def control_loop_callback(self):
        try:
            with self.data_lock:
                marker_detected = self.marker_data.detected
                marker_center = self.marker_data.center
                marker_area = self.marker_data.area
                current_time = time.time()
            
            # State machine
            if self.state == ControllerState.INITIALIZING:
                self.state = ControllerState.WAITING_FOR_MARKER
                self.state_entry_time = current_time
                
            elif self.state == ControllerState.WAITING_FOR_MARKER:
                if marker_detected:
                    self.get_logger().info("Marker detected! Starting centering phase.")
                    self.state = ControllerState.CENTERING
                    self.state_entry_time = current_time
                
            elif self.state == ControllerState.CENTERING:
                if not marker_detected:
                    if current_time - self.last_marker_seen_time > self.marker_lost_timeout:
                        self.get_logger().warn("Marker lost during centering. Waiting for marker.")
                        self.state = ControllerState.WAITING_FOR_MARKER
                else:
                    # Compute centering error
                    offset_x = marker_center[0] - self.image_center_x
                    offset_y = marker_center[1] - self.image_center_y
                    
                    # Proportional controller: normalize by image center
                    vel_x = -self.kp_x * (offset_x / self.image_center_x)
                    vel_y = self.kp_y * (offset_y / self.image_center_y)
                    
                    # Clamp velocities
                    vel_x = np.clip(vel_x, -self.max_horizontal_vel, self.max_horizontal_vel)
                    vel_y = np.clip(vel_y, -self.max_horizontal_vel, self.max_horizontal_vel)
                    
                    # Check if centered
                    is_centered = (abs(offset_x) < self.centering_tolerance and 
                                 abs(offset_y) < self.centering_tolerance)
                    
                    if is_centered or (current_time - self.state_entry_time > self.centering_timeout):
                        self.get_logger().info("Marker centered! Starting descent.")
                        self.state = ControllerState.DESCENDING
                        self.state_entry_time = current_time
                    else:
                        # Continue centering
                        self.velocity_cmd.linear.x = vel_x
                        self.velocity_cmd.linear.y = vel_y
                        self.velocity_cmd.linear.z = 0.0
                        self.velocity_pub.publish(self.velocity_cmd)
                
            elif self.state == ControllerState.DESCENDING:
                if marker_area > self.max_marker_area:
                    self.get_logger().info(f"Marker too close (area: {marker_area}px). Landing.")
                    self.state = ControllerState.LANDING
                    self.state_entry_time = current_time
                elif not marker_detected:
                    if current_time - self.last_marker_seen_time > self.marker_lost_timeout:
                        self.get_logger().info("Marker lost during descent. Landing.")
                        self.state = ControllerState.LANDING
                        self.state_entry_time = current_time
                elif current_time - self.state_entry_time > self.descent_timeout:
                    self.get_logger().warn("Descent timeout. Landing.")
                    self.state = ControllerState.LANDING
                    self.state_entry_time = current_time
                else:
                    # Soft centering while descending (lighter gains)
                    offset_x = marker_center[0] - self.image_center_x
                    offset_y = marker_center[1] - self.image_center_y
                    
                    vel_x = -0.5 * self.kp_x * (offset_x / self.image_center_x)
                    vel_y = 0.5 * self.kp_y * (offset_y / self.image_center_y)
                    
                    vel_x = np.clip(vel_x, -self.max_horizontal_vel, self.max_horizontal_vel)
                    vel_y = np.clip(vel_y, -self.max_horizontal_vel, self.max_horizontal_vel)
                    
                    # Descend
                    self.velocity_cmd.linear.x = vel_x
                    self.velocity_cmd.linear.y = vel_y
                    self.velocity_cmd.linear.z = np.clip(self.descent_velocity, -self.max_descent_vel, 0.0)
                    self.velocity_pub.publish(self.velocity_cmd)
                
            elif self.state == ControllerState.LANDING:
                # Send LAND action
                self._send_land_action()
                self.state = ControllerState.COMPLETED
                
            elif self.state == ControllerState.COMPLETED:
                self.get_logger().info("Mission completed. Shutting down.")
                self.control_timer.cancel()
                
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.state = ControllerState.ERROR

    def _send_land_action(self):
        try:
            self.get_logger().info("Sending LAND action.")
            
            if self.land_client is None:
                self.get_logger().warn('LAND action disabled. Trying MAVROS land service fallback.')
                self.velocity_cmd.linear.x = 0.0
                self.velocity_cmd.linear.y = 0.0
                self.velocity_cmd.linear.z = 0.0
                self.velocity_pub.publish(self.velocity_cmd)
                self._send_mavros_land_service()
                return

            if not self.land_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn("Land action server not available. Publishing zero velocity instead.")
                self.velocity_cmd.linear.x = 0.0
                self.velocity_cmd.linear.y = 0.0
                self.velocity_cmd.linear.z = 0.0
                self.velocity_pub.publish(self.velocity_cmd)
                return
            
            goal = Land.Goal()
            goal.landing_altitude = 0.0
            
            future = self.land_client.send_goal_async(goal)
            future.add_done_callback(self._land_goal_response_callback)
            
        except Exception as e:
            self.get_logger().error(f"Error sending land action: {e}")

    def _send_mavros_land_service(self):
        if self.mavros_land_client is None:
            self.get_logger().warn('MAVROS land service disabled.')
            return
        if not self.mavros_land_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('MAVROS land service not available.')
            return

        request = CommandTOL.Request()
        request.min_pitch = 0.0
        request.yaw = 0.0
        request.latitude = 0.0
        request.longitude = 0.0
        request.altitude = 0.0
        future = self.mavros_land_client.call_async(request)
        future.add_done_callback(self._mavros_land_response_callback)

    def _mavros_land_response_callback(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"MAVROS land service call failed: {e}")
            return

        if response.success:
            self.get_logger().info('MAVROS land service accepted.')
        else:
            self.get_logger().warn(f"MAVROS land service rejected with result: {response.result}")

    def _land_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Land goal rejected.")
        else:
            self.get_logger().info("Land goal accepted.")


def main(args=None):
    rclpy.init(args=args)
    
    controller_node = ArucoCenteredDescentController()
    executor = MultiThreadedExecutor()
    executor.add_node(controller_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        controller_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
