import argparse
import socket
import time

import numpy as np
import zerorpc

from droid.misc.parameters import nuc_ip
from droid.misc.server_interface import ServerInterface


def check_tcp_port(host, port, timeout):
    with socket.create_connection((host, port), timeout=timeout):
        return True


def print_state(state_dict):
    print("joint_positions:", np.round(np.array(state_dict["joint_positions"]), 4).tolist())
    print("joint_velocities:", np.round(np.array(state_dict["joint_velocities"]), 4).tolist())
    print("cartesian_position:", np.round(np.array(state_dict["cartesian_position"]), 4).tolist())
    print("gripper_position:", round(float(state_dict["gripper_position"]), 4))
    print("prev_command_successful:", state_dict.get("prev_command_successful"))


def get_robot_state_or_explain(robot):
    try:
        return robot.get_robot_state()
    except zerorpc.exceptions.RemoteError as err:
        if "_robot" in str(err):
            raise SystemExit(
                "Connected to the NUC ZeroRPC server, but the Franka robot client is not launched yet.\n"
                "Run again with --launch after enabling FCI/unlocking the robot in Franka Desk:\n"
                "  python scripts/tests/check_nuc_control.py --host <NUC_IP> --launch"
            )
        raise


def main():
    parser = argparse.ArgumentParser(description="Check DROID control laptop -> NUC -> Franka communication.")
    parser.add_argument("--host", default=nuc_ip, help="NUC/control-server IP address.")
    parser.add_argument("--port", type=int, default=4242, help="DROID ZeroRPC server port.")
    parser.add_argument("--timeout", type=float, default=3.0, help="TCP connection timeout in seconds.")
    parser.add_argument("--launch", action="store_true", help="Ask the NUC server to launch Polymetis and the robot.")
    parser.add_argument("--test-motion", action="store_true", help="Move joint 7 by a small delta and then move back.")
    parser.add_argument("--motion-delta", type=float, default=0.02, help="Joint 7 test delta in radians.")
    parser.add_argument(
        "--test-velocity",
        action="store_true",
        help="Run a small OpenPI-style joint_velocity command and then command zero velocity.",
    )
    parser.add_argument("--velocity-steps", type=int, default=5, help="Number of velocity test command steps.")
    parser.add_argument("--velocity-value", type=float, default=0.02, help="Normalized joint 7 velocity test command.")
    parser.add_argument("--test-gripper", action="store_true", help="Open and close the gripper slightly.")
    args = parser.parse_args()

    print("Checking TCP connectivity to {}:{}...".format(args.host, args.port))
    check_tcp_port(args.host, args.port, args.timeout)
    print("TCP port is reachable.")

    print("Connecting to DROID ZeroRPC server...")
    robot = ServerInterface(ip_address=args.host, launch=False)
    print("ZeroRPC connection created.")

    if args.launch:
        print("Launching controller and robot on the NUC. This may take a few seconds...")
        robot.launch_controller()
        robot.launch_robot()
        print("Robot launched.")

    print("Reading robot state...")
    state_dict, timestamp_dict = get_robot_state_or_explain(robot)
    print_state(state_dict)
    print("robot_timestamp_seconds:", timestamp_dict.get("robot_timestamp_seconds"))

    if args.test_gripper:
        current = float(state_dict["gripper_position"])
        target = float(np.clip(current + 0.1, 0.0, 1.0))
        print("Moving gripper from {:.3f} to {:.3f} and back...".format(current, target))
        robot.update_gripper(target, velocity=False, blocking=True)
        time.sleep(0.5)
        robot.update_gripper(current, velocity=False, blocking=True)
        print("Gripper test complete.")

    if args.test_motion:
        current_joints = np.array(state_dict["joint_positions"], dtype=float)
        target_joints = current_joints.copy()
        target_joints[-1] += args.motion_delta

        print("Moving joint 7 by {:.4f} rad and then returning...".format(args.motion_delta))
        robot.update_joints(target_joints, velocity=False, blocking=True)
        time.sleep(0.5)
        robot.update_joints(current_joints, velocity=False, blocking=True)
        print("Motion test complete.")

    if args.test_velocity:
        action = np.zeros(8)
        action[-2] = args.velocity_value
        zero_action = np.zeros(8)

        print(
            "Running OpenPI-style joint_velocity test: joint 7 command {:.4f} for {} steps...".format(
                args.velocity_value, args.velocity_steps
            )
        )
        for _ in range(args.velocity_steps):
            robot.update_command(action, action_space="joint_velocity", gripper_action_space="position", blocking=False)
            time.sleep(1 / 15)
        for _ in range(3):
            robot.update_command(
                zero_action, action_space="joint_velocity", gripper_action_space="position", blocking=False
            )
            time.sleep(1 / 15)
        print("Velocity test complete.")

    print("Final robot state:")
    final_state, _ = get_robot_state_or_explain(robot)
    print_state(final_state)


if __name__ == "__main__":
    main()
