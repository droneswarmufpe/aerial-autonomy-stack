from pymavlink import mavutil
import argparse
import time

def set_message_interval(master, message_id, frequency_hz):
    """
    Set MAVLink message interval using MAV_CMD_SET_MESSAGE_INTERVAL
    """

    if frequency_hz <= 0:
        interval_us = -1
    else:
        interval_us = int(1e6 / frequency_hz)

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0
    )

def request_data_stream(master, target_system, target_component, stream_id, rate, start_stop):
    """
    Legacy REQUEST_DATA_STREAM
    """

    master.mav.request_data_stream_send(
        target_system,
        target_component,
        stream_id,
        rate,
        start_stop
    )

def main():
    parser = argparse.ArgumentParser(
        description='Configure MAVLink telemetry streams'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='/dev/ttyTHS1'
    )

    parser.add_argument(
        '--baudrate',
        type=int,
        default=921600
    )

    parser.add_argument(
        '--target-system',
        type=int,
        default=1
    )

    parser.add_argument(
        '--target-component',
        type=int,
        default=1
    )

    parser.add_argument(
        '--rate',
        type=float,
        default=10
    )

    parser.add_argument(
        '--streamrate',
        type=int,
        default=-1,
        help='Legacy REQUEST_DATA_STREAM rate. -1 disables stream requests.'
    )

    parser.add_argument(
        '--use-message-interval',
        action='store_true',
        help='Use MAV_CMD_SET_MESSAGE_INTERVAL'
    )

    args = parser.parse_args()

    print(f"Connecting to {args.device} @ {args.baudrate}")

    master = mavutil.mavlink_connection(
        args.device,
        baud=args.baudrate
    )

    master.wait_heartbeat()

    print(
        f"Heartbeat received from "
        f"system={master.target_system} "
        f"component={master.target_component}"
    )

    #
    # Legacy REQUEST_DATA_STREAM handling
    #

    streams = [
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
    ]

    if args.streamrate == -1:
        print("Disabling legacy REQUEST_DATA_STREAM")

        for stream in streams:
            request_data_stream(
                master,
                args.target_system,
                args.target_component,
                stream,
                0,
                0
            )

    else:
        print(
            f"Requesting legacy streams at "
            f"{args.streamrate} Hz"
        )

        for stream in streams:
            request_data_stream(
                master,
                args.target_system,
                args.target_component,
                stream,
                args.streamrate,
                1
            )

    #
    # Modern SET_MESSAGE_INTERVAL
    #

    if args.use_message_interval:

        print(
            f"Setting message intervals to "
            f"{args.rate} Hz"
        )

        message_rates = {
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT: 1,
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS: args.rate,
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: args.rate,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: args.rate,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: args.rate,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: args.rate,
        }

        for message_id, hz in message_rates.items():
            set_message_interval(master, message_id, hz)
            time.sleep(0.05)

    print("Telemetry configuration complete")

if __name__ == "__main__":
    main()