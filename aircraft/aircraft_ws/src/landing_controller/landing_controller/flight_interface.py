from geometry_msgs.msg import Twist
from mavros_msgs.srv import CommandTOL
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


class FlightInterface:
    def __init__(self, node, command_topic: str, setpoint_velocity_node: str, velocity_frame: str):
        self.node = node
        self.velocity_frame = velocity_frame
        self.frame_request_pending = False
        self.latest_command = Twist()

        self.velocity_pub = node.create_publisher(Twist, command_topic, 10)
        self.land_client = node.create_client(CommandTOL, '/mavros/cmd/land')
        self.param_client = node.create_client(SetParameters, f'{setpoint_velocity_node}/set_parameters')
        self.frame_timer = node.create_timer(1.0, self._set_velocity_frame)

    def set_velocity(self, forward: float, right: float, down: float):
        cmd = Twist()
        cmd.linear.x = forward
        cmd.linear.y = right
        cmd.linear.z = down
        self.latest_command = cmd
        self.velocity_pub.publish(cmd)

    def command_text(self) -> str:
        cmd = self.latest_command.linear
        return f'cmd x={cmd.x:.2f} y={cmd.y:.2f} z={cmd.z:.2f}'

    def land(self):
        if not self.land_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().warn('/mavros/cmd/land is not available.')
            return

        request = CommandTOL.Request()
        request.min_pitch = 0.0
        request.yaw = 0.0
        request.latitude = 0.0
        request.longitude = 0.0
        request.altitude = 0.0
        self.land_client.call_async(request).add_done_callback(self._land_response)

    def _land_response(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.node.get_logger().error(f'LAND service failed: {exc}')
            return
        if response.success:
            self.node.get_logger().info('LAND accepted.')
        else:
            self.node.get_logger().warn(f'LAND rejected: result={response.result}')

    def _set_velocity_frame(self):
        if self.frame_request_pending:
            return
        if not self.velocity_frame:
            self.frame_timer.cancel()
            return
        if not self.param_client.service_is_ready():
            return

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name='mav_frame',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=self.velocity_frame,
                ),
            )
        ]
        self.frame_request_pending = True
        self.param_client.call_async(request).add_done_callback(self._velocity_frame_response)

    def _velocity_frame_response(self, future):
        self.frame_request_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.node.get_logger().warn(f'Could not set MAVROS velocity frame: {exc}')
            return
        if response.results and response.results[0].successful:
            self.node.get_logger().info(f'MAVROS velocity frame set to {self.velocity_frame}')
            self.frame_timer.cancel()
        else:
            reason = response.results[0].reason if response.results else 'no response'
            self.node.get_logger().warn(f'MAVROS velocity frame was not set: {reason}')
