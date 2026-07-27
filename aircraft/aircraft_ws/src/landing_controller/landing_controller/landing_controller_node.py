import threading
import time
from enum import Enum

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from landing_controller.flight_interface import FlightInterface
from landing_controller.marker_vision import (
    DebugVideo,
    GStreamerCamera,
    MarkerObservation,
    MarkerTracker,
)


SHOW_CONTROLLER_VIEW_VIDEO = False


class ControllerState(Enum):
    WAITING_FOR_MARKER = 'waiting'
    CENTERING = 'centering'
    DESCENDING = 'descending'
    LANDING = 'landing'
    DONE = 'done'


class LandingController(Node):
    def __init__(self):
        super().__init__('landing_controller')
        self._declare_params()
        self._load_params()
        self._init_state()

        self.bridge = CvBridge()
        self.tracker = MarkerTracker(self.marker_id, self.min_marker_area_px)
        self.flight = FlightInterface(
            self,
            self.command_topic,
            self.mavros_setpoint_velocity_node,
            self.mavros_velocity_frame,
        )

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Image, self.camera_topic, self._camera_callback, qos)
        self.debug_pub = (
            self.create_publisher(Image, 'landing_controller/debug_image', 10)
            if self.publish_debug_images else None
        )

        self.video = DebugVideo(SHOW_CONTROLLER_VIEW_VIDEO, 'landing_controller', self.get_logger())
        self.video.open()
        self.camera = GStreamerCamera(self.camera_stream_port, self.get_logger())
        self.camera.start(self._handle_frame)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f'landing_controller ready: marker={self.marker_id}, '
            f'camera_udp={self.camera_stream_port}, frame={self.mavros_velocity_frame}'
        )

    def destroy_node(self):
        self.camera.stop()
        self.video.close()
        super().destroy_node()

    def _declare_params(self):
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('camera_stream_port', 5600)
        self.declare_parameter('command_topic', '/mavros/setpoint_velocity/cmd_vel_unstamped')
        self.declare_parameter('mavros_setpoint_velocity_node', '/mavros/setpoint_velocity')
        self.declare_parameter('mavros_velocity_frame', 'BODY_NED')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('kp_forward', 0.35)
        self.declare_parameter('kp_right', 0.35)
        self.declare_parameter('descent_velocity', -0.20)
        self.declare_parameter('max_horizontal_velocity', 0.45)
        self.declare_parameter('centering_tolerance_px', 14)
        self.declare_parameter('center_hold_sec', 0.6)
        self.declare_parameter('min_marker_area_px', 80)
        self.declare_parameter('land_marker_area_px', 14000)
        self.declare_parameter('marker_lost_timeout_sec', 1.5)
        self.declare_parameter('centering_timeout_sec', 15.0)
        self.declare_parameter('descent_timeout_sec', 60.0)
        self.declare_parameter('publish_debug_images', False)

    def _load_params(self):
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_stream_port = int(self.get_parameter('camera_stream_port').value)
        self.command_topic = self.get_parameter('command_topic').value
        self.mavros_setpoint_velocity_node = self.get_parameter('mavros_setpoint_velocity_node').value
        self.mavros_velocity_frame = self.get_parameter('mavros_velocity_frame').value
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.kp_forward = float(self.get_parameter('kp_forward').value)
        self.kp_right = float(self.get_parameter('kp_right').value)
        self.descent_velocity = float(self.get_parameter('descent_velocity').value)
        self.max_horizontal_velocity = float(self.get_parameter('max_horizontal_velocity').value)
        self.centering_tolerance_px = float(self.get_parameter('centering_tolerance_px').value)
        self.center_hold_sec = float(self.get_parameter('center_hold_sec').value)
        self.min_marker_area_px = float(self.get_parameter('min_marker_area_px').value)
        self.land_marker_area_px = float(self.get_parameter('land_marker_area_px').value)
        self.marker_lost_timeout_sec = float(self.get_parameter('marker_lost_timeout_sec').value)
        self.centering_timeout_sec = float(self.get_parameter('centering_timeout_sec').value)
        self.descent_timeout_sec = float(self.get_parameter('descent_timeout_sec').value)
        self.publish_debug_images = bool(self.get_parameter('publish_debug_images').value)

    def _init_state(self):
        self.state = ControllerState.WAITING_FOR_MARKER
        self.state_started_at = time.time()
        self.center_started_at = None
        self.marker = MarkerObservation()
        self.image_width = 320
        self.image_height = 240
        self.lock = threading.Lock()

    def _camera_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Camera frame conversion failed: {exc}')
            return
        self._handle_frame(frame)

    def _handle_frame(self, frame):
        marker = self.tracker.detect(frame)
        with self.lock:
            self.image_height, self.image_width = frame.shape[:2]
            if not marker.detected:
                marker.seen_at = self.marker.seen_at
            self.marker = marker

        if self.debug_pub is not None or self.video.opened:
            debug = self.tracker.draw(frame.copy(), marker, self.state.value, self.flight.command_text())
            if self.debug_pub is not None:
                msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                self.debug_pub.publish(msg)
            self.video.show(debug)

    def _control_loop(self):
        now = time.time()
        with self.lock:
            marker = self.marker
            width = max(float(self.image_width), 1.0)
            height = max(float(self.image_height), 1.0)

        if self.state == ControllerState.WAITING_FOR_MARKER:
            self.flight.set_velocity(0.0, 0.0, 0.0)
            if marker.detected:
                self._enter_state(ControllerState.CENTERING)
            return

        if self.state == ControllerState.CENTERING:
            self._center(marker, now, width, height)
            return

        if self.state == ControllerState.DESCENDING:
            self._descend(marker, now, width, height)
            return

        if self.state == ControllerState.LANDING:
            self.flight.set_velocity(0.0, 0.0, 0.0)
            self.flight.land()
            self._enter_state(ControllerState.DONE)
            return

        if self.state == ControllerState.DONE:
            self.flight.set_velocity(0.0, 0.0, 0.0)

    def _center(self, marker: MarkerObservation, now: float, width: float, height: float):
        if not marker.detected:
            self.flight.set_velocity(0.0, 0.0, 0.0)
            if not self._marker_seen_recently(marker, now):
                self.center_started_at = None
                self._enter_state(ControllerState.WAITING_FOR_MARKER)
            return

        forward, right, centered = self._centering_command(marker, width, height, descent=False)
        self.flight.set_velocity(forward, right, 0.0)

        if centered:
            if self.center_started_at is None:
                self.center_started_at = now
            if now - self.center_started_at >= self.center_hold_sec:
                self._enter_state(ControllerState.DESCENDING)
            return

        self.center_started_at = None
        if now - self.state_started_at > self.centering_timeout_sec:
            self.get_logger().warn('Centering timeout; holding position and waiting for marker.')
            self.flight.set_velocity(0.0, 0.0, 0.0)
            self._enter_state(ControllerState.WAITING_FOR_MARKER)

    def _descend(self, marker: MarkerObservation, now: float, width: float, height: float):
        if not marker.detected:
            self.flight.set_velocity(0.0, 0.0, 0.0)
            if not self._marker_seen_recently(marker, now):
                self.get_logger().warn('Marker lost during descent; holding position instead of landing blind.')
                self._enter_state(ControllerState.WAITING_FOR_MARKER)
            return

        if now - self.state_started_at > self.descent_timeout_sec:
            self.get_logger().warn('Descent timeout; holding position instead of landing blind.')
            self.flight.set_velocity(0.0, 0.0, 0.0)
            self._enter_state(ControllerState.WAITING_FOR_MARKER)
            return

        forward, right, centered = self._centering_command(marker, width, height, descent=True)
        if marker.area >= self.land_marker_area_px:
            if centered:
                self.get_logger().info(f'Marker is close and centered: area={marker.area:.0f}px')
                self._enter_state(ControllerState.LANDING)
            else:
                self.flight.set_velocity(forward, right, 0.0)
            return

        self.flight.set_velocity(forward, right, self.descent_velocity)

    def _centering_command(self, marker: MarkerObservation, width: float, height: float, descent: bool):
        center_x = width / 2.0
        center_y = height / 2.0
        error_x = (marker.center_x - center_x) / center_x
        error_y = (marker.center_y - center_y) / center_y

        gain_scale = 0.7 if descent else 1.0
        forward = -self.kp_forward * error_y * gain_scale
        right = -self.kp_right * error_x * gain_scale
        forward = float(np.clip(forward, -self.max_horizontal_velocity, self.max_horizontal_velocity))
        right = float(np.clip(right, -self.max_horizontal_velocity, self.max_horizontal_velocity))

        centered = (
            abs(marker.center_x - center_x) <= self.centering_tolerance_px
            and abs(marker.center_y - center_y) <= self.centering_tolerance_px
        )
        return forward, right, centered

    def _marker_seen_recently(self, marker: MarkerObservation, now: float) -> bool:
        return marker.detected or (now - marker.seen_at <= self.marker_lost_timeout_sec)

    def _enter_state(self, state: ControllerState):
        if self.state == state:
            return
        self.state = state
        self.state_started_at = time.time()
        if state != ControllerState.CENTERING:
            self.center_started_at = None
        self.get_logger().info(f'State: {state.value}')


def main(args=None):
    rclpy.init(args=args)
    node = LandingController()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
